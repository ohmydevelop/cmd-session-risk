/* cmd-session-risk intent layer (deterministic, alert-only).
 *
 * This is a mini shell-semantics analyzer used as a CERTIFICATION backstop
 * over the BERT detector.  It can never replace the model for generalization:
 * arbitrary Turing-complete obfuscation (perl -e, python -c, VM-based) can
 * always defeat a static analyzer; a behavioral layer (eBPF) is the only
 * airtight defense.  What this layer does provide is:
 *
 *   - statement / pipeline splitting             (; && || |)
 *   - sh -c / bash -c payload unwrapping        (recursive)
 *   - active evaluation of common builtins:      echo, printf, base64 -d
 *   - ANSI-C quotes, \xNN / \NNN escapes, ${IFS}
 *   - lexical path canonicalization              (/tmp/../etc -> /etc)
 *   - semantic intent predicates (destructive-delete-of-system-path,
 *     raw-device-write, download-and-execute, reverse shell, uid-0 backdoor)
 *
 * Output semantics: hit => decision is FORCED to 1 (alert).  It never forces
 * 0 and never blocks anything.  Column 5 reports the hit.
 *
 * v1.3.0 (2026-09-23) supersedes the v1.1.2 substring guard; every v1.1.2
 * rule is preserved as a predicate of this analyzer.
 */
#ifndef CMDRISK_INTENT_H
#define CMDRISK_INTENT_H

#include <string.h>

static int chr_lc(char a){ return (a>='A' && a<='Z') ? (a-'A'+'a') : a; }
static int is_hexc(char c){ return (c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F'); }
static int hex_val(char c){
    if(c>='0'&&c<='9') return c-'0';
    c = chr_lc(c);
    return (c>='a' && c<='f') ? (c-'a'+10) : 0;
}
static int is_octc(char c){ return c>='0' && c<='7'; }
static int is_alnum(char c){ return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9'); }

static int has_sub(const char *hay, const char *needle){
    if(!needle[0]) return 1;
    for(; *hay; hay++){
        const char *h=hay, *n=needle;
        while(*h && *n){ if(chr_lc(*h)!=chr_lc(*n)) break; h++; n++; }
        if(!*n) return 1;
    }
    return 0;
}

/* forward */
static int intent_statement(const char *s, int depth);

/* ------------------------------------------------------------------ */
/* escape decoding: \xNN (hex), \NNN (octal, posix printf)             */
/* returns new length starting at q; writes into dst                   */
/* ------------------------------------------------------------------ */
static const char *decode_escapes_into(const char *p, char *dst, size_t cap){
    size_t j=0;
    while(*p && *p!='\n' && j+1<cap){
        if(*p=='\\' && p[1]){
            char n=p[1];
            if(n=='x' && is_hexc(p[2]) && is_hexc(p[3])){
                dst[j++]=(char)(hex_val(p[2])*16+hex_val(p[3])); p+=4; continue;
            }
            if(n>='0' && n<='7'){
                int v=0, k=0;
                while(k<3 && is_octc(p[1])){ v=v*8+(p[1]-'0'); p++; k++; }
                dst[j++]=(char)(v&0xff); continue;
            }
            if(n=='n'){ dst[j++]='\n'; p+=2; continue; }
            if(n=='t'){ dst[j++]='\t'; p+=2; continue; }
            if(n=='r'){ dst[j++]='\r'; p+=2; continue; }
            dst[j++]=n; p+=2; continue;
        }
        dst[j++]=*p; p++;
    }
    dst[j]=0;
    return p;
}

static const char *decode_escapes_until(const char *p, char *dst, size_t cap, const char *stop){
    size_t j=0;
    while(*p && !strchr(stop,*p) && j+1<cap){
        if(*p=='\\' && p[1]){
            char n=p[1];
            if(n=='x' && is_hexc(p[2]) && is_hexc(p[3])){
                dst[j++]=(char)(hex_val(p[2])*16+hex_val(p[3])); p+=4; continue;
            }
            if(n>='0' && n<='7'){
                int v=0, k=0;
                while(k<3 && is_octc(p[1])){ v=v*8+(p[1]-'0'); p++; k++; }
                dst[j++]=(char)(v&0xff); continue;
            }
            if(n=='n'){ dst[j++]='\n'; p+=2; continue; }
            if(n=='t'){ dst[j++]='\t'; p+=2; continue; }
            if(n=='r'){ dst[j++]='\r'; p+=2; continue; }
            dst[j++]=n; p+=2; continue;
        }
        dst[j++]=*p; p++;
    }
    dst[j]=0;
    (void)cap;
    return p;
}

/* ------------------------------------------------------------------ */
/* deobfuscate: quotes, escapes, trivial substitutions, whitespace     */
/* ------------------------------------------------------------------ */
static void deobfuscate(const char *src, char *dst, size_t cap){
    size_t j=0;
    const char *p=src;
    while(*p && j+1<cap){
        char c=*p;
        if(c=='\\'){
            if(p[1]=='x' && is_hexc(p[2]) && is_hexc(p[3])){
                dst[j++]=(char)(hex_val(p[2])*16+hex_val(p[3])); p+=4; continue;
            }
            if(p[1]>='0' && p[1]<='7'){
                int v=0,k=0;
                while(k<3 && is_octc(p[1])){ v=v*8+(p[1]-'0'); p++; k++; }
                dst[j++]=(char)(v&0xff); continue;
            }
            if(p[1]){ dst[j++]=(char)p[1]; p+=2; continue; }
            p++; continue;
        }
        if(c=='\'' || c=='"'){ p++; continue; }
        if(c=='$' && p[1]=='\''){           /* ANSI-C quote $'...' */
            p+=2; const char *q=p;
            while(*q && *q!='\''){
                if(*q=='\\' && q[1]=='\'') q+=2; else q++;
            }
            char tmp[512];
            const char *q2=p;
            size_t k=0;
            while(q2<q && k+1<sizeof(tmp)){
                if(*q2=='\\' && q2+1<q){
                    char n=q2[1];
                    if(n=='x' && q2+3<q && is_hexc(q2[2]) && is_hexc(q2[3])){
                        tmp[k++]=(char)(hex_val(q2[2])*16+hex_val(q2[3])); q2+=4; continue;
                    }
                    if(n>='0'&&n<='7'){ int v=0,kk=0; while(kk<3 && q2+1<q && is_octc(q2[1])){ v=v*8+(q2[1]-'0'); q2++; kk++; } tmp[k++]=(char)(v&0xff); continue; }
                    if(n=='n'){ tmp[k++]='\n'; q2+=2; continue; }
                    if(n=='t'){ tmp[k++]='\t'; q2+=2; continue; }
                    tmp[k++]=n; q2+=2; continue;
                }
                tmp[k++]=*q2++;
            }
            tmp[k]=0;
            for(size_t z=0;tmp[z] && j+1<cap;z++) dst[j++]=tmp[z];
            p = (*q=='\'') ? q+1 : q;
            continue;
        }
        if(c=='$' && (p[1]=='(' || p[1]=='{')){
            if(p[1]=='(' ){
                const char *q=p+2; int depth=0;
                while(*q && !(*q==')' && depth==0)){
                    if(*q=='(') depth++; if(*q==')') depth--;
                    q++;
                }
                const char *e=p+2;
                while(e<q && (*e==' '||*e=='\t')) e++;
                const char *w=e; while(w<q && *w!=' ' && *w!='\t') w++;
                if(w>e && w-e==4 && chr_lc(e[0])=='e' && chr_lc(e[1])=='c'
                   && chr_lc(e[2])=='h' && chr_lc(e[3])=='o'){
                    while(w<q && (*w==' '||*w=='\t')) w++;
                    if(w<q){ char tmp[512]; decode_escapes_until(w,tmp,sizeof tmp,")");
                        for(size_t z=0;tmp[z] && j+1<cap;z++) dst[j++]=tmp[z]; }
                } else if(w>e && w-e==6 && chr_lc(e[0])=='p' && chr_lc(e[1])=='r'
                   && chr_lc(e[2])=='i' && chr_lc(e[3])=='n' && chr_lc(e[4])=='t' && chr_lc(e[5])=='f'){
                    /* consume format arg, decode following arg */
                    while(w<q && (*w==' '||*w=='\t')) w++;
                    while(w<q && *w!=' ' && *w!='\t') w++;
                    while(w<q && (*w==' '||*w=='\t')) w++;
                    if(w<q){ char tmp[512]; decode_escapes_until(w,tmp,sizeof tmp,")");
                        for(size_t z=0;tmp[z] && j+1<cap;z++) dst[j++]=tmp[z]; }
                } else if(j+1<cap){ dst[j++]='?'; }
                if(*q==')') q++;
                p=q; continue;
            } else { /* ${...} */
                const char *q=p+2; while(*q && *q!='}') q++;
                if((q-(p+2))==3 && chr_lc(p[2])=='i' && chr_lc(p[3])=='f' && chr_lc(p[4])=='s'){
                    if(j && dst[j-1]!=' ') dst[j++]=' ';
                } else if(j+1<cap){ if(j+3<cap){ dst[j++]='$'; dst[j++]='{'; for(const char*z=p+2; z<q && j+2<cap; z++) dst[j++]=*z; dst[j++]='}'; } }
                p = (*q=='}') ? q+1 : q;
                continue;
            }
        }
        if(c=='`'){
            const char *q=p+1; while(*q && *q!='`') q++;
            const char *e=p+1;
            while(e<q && (*e==' '||*e=='\t')) e++;
            const char *w=e; while(w<q && *w!=' ' && *w!='\t') w++;
            if(w>e && w-e==4 && chr_lc(e[0])=='e' && chr_lc(e[1])=='c'
               && chr_lc(e[2])=='h' && chr_lc(e[3])=='o'){
                while(w<q && (*w==' '||*w=='\t')) w++;
                if(w<q){ char tmp[512]; decode_escapes_until(w,tmp,sizeof tmp,"`");
                    for(size_t z=0;tmp[z] && j+1<cap;z++) dst[j++]=tmp[z]; }
            } else if(j+1<cap){ dst[j++]='?'; }
            if(*q=='`') q++;
            p=q; continue;
        }
        if(c==' '||c=='\t'||c=='\r'||c=='\n'){
            if(j && dst[j-1]!=' ') dst[j++]=' ';
            p++; continue;
        }
        dst[j++]=c; p++;
    }
    dst[j]=0;
    /* collapse "//" and "/./" (lexical, does not resolve ".."; that is done
       per-token in canonicalize_path) */
    char out[560]; size_t k=0; const char *s=dst;
    while(*s && k+1<sizeof(out)){
        if(*s=='/'){
            out[k++]='/'; s++;
            while(*s=='/') s++;
            while(s[0]=='.' && s[1]=='/') s+=2;
            continue;
        }
        out[k++]=*s++;
    }
    out[k]=0;
    if(k+1<=cap){ size_t i=0; for(;out[i];i++) dst[i]=out[i]; dst[i]=0; }
}

/* lexical canonicalization of an ABSOLUTE POSIX path */
static void canonicalize_path(const char *in, char *out, size_t cap){
    char stack[64][256]; int sp=0; char tok[256];
    const char *p=in; size_t o=0;
    if(*p!='/') return;
    while(*p){
        if(*p=='/'){ const char *s=++p; size_t t=0;
            while(*p && *p!='/'){ if(t+1<sizeof(tok)) tok[t++]=*p; p++; }
            tok[t]=0;
            if(!tok[0]) continue;
            if(!strcmp(tok,".")) continue;
            if(!strcmp(tok,"..")){ if(sp>0) sp--; continue; }
            if(sp<64){ strcpy(stack[sp++],tok); }
        } else p++;
    }
    if(sp==0){ snprintf(out,cap,"/"); return; }
    for(int i=0;i<sp;i++){ o+= (size_t)snprintf(out+o, o<cap? cap-o:1, "/%s", stack[i]); if(o>=cap){out[cap-1]=0;return;} }
    out[o]=0;
}

/* standard base64 decode; returns 1 on success */
static int base64_decode(const char *src, char *out, size_t cap){
    static const char *tab="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int vals[256]; memset(vals,-1,sizeof vals);
    for(int i=0;tab[i];i++) vals[(unsigned char)tab[i]]=i;
    size_t o=0; int buf=0, bits=0;
    for(const char *p=src; *p; p++){
        if(*p==' '||*p=='\t'||*p=='\r'||*p=='\n') continue;
        int v=vals[(unsigned char)*p];
        if(v<0) return 0;
        buf=(buf<<6)|v; bits+=6;
        if(bits>=8){ bits-=8; if(o+1<cap) out[o++]=(char)((buf>>bits)&0xff); }
    }
    out[o]=0;
    return 1;
}

/* ---------------------------------------------------------------- */
/* token helpers                                                     */
/* ---------------------------------------------------------------- */
static int stage_first_word(const char *stage, const char *w){ /* first non-space token */
    while(*stage==' ') stage++;
    size_t n=strlen(w);
    if(stage[n]!=0 && stage[n]!=' ') return 0;
    for(size_t i=0;i<n;i++) if(chr_lc(stage[i])!=chr_lc(w[i])) return 0;
    return 1;
}

static int token_is_in(const char *stage, const char *w){
    const char *p=stage;
    while(*p){
        while(*p==' ') p++;
        const char *s=p; while(*p && *p!=' ' && *p!='|') p++;
        size_t n=(size_t)(p-s);
        if(n==strlen(w)){ size_t i=0; for(;i<n;i++) if(chr_lc(s[i])!=chr_lc(w[i])) break; if(i==n) return 1; }
        while(*p==' ') p++;
    }
    return 0;
}

static int device_path(const char *tok){
    return has_sub(tok,"/dev/sd") || has_sub(tok,"/dev/hd") || has_sub(tok,"/dev/nvme")
        || has_sub(tok,"/dev/mmcblk") || has_sub(tok,"/dev/mapper")
        || has_sub(tok,"/dev/disk") || has_sub(tok,"/dev/md")
        || has_sub(tok,"/dev/vd") || has_sub(tok,"/dev/xvd") || has_sub(tok,"/dev/zd");
}

/* destructive path predicate */
static int path_danger(const char *tok){
    int star=0; size_t n=strlen(tok);
    if(n>=2 && tok[n-1]=='*' && tok[n-2]=='/'){ star=1; n-=2; }
    if(n==0) return 0;
    char raw[520]; size_t k=0;
    for(size_t i=0;i<n && k+1<sizeof(raw);i++) raw[k++]=(char)chr_lc(tok[i]);
    raw[k]=0;
    char path[520];
    if(raw[0]=='/'){ canonicalize_path(raw,path,sizeof path); }
    else {
        int home = (chr_lc(raw[0])=='$' && chr_lc(raw[1])=='h' && chr_lc(raw[2])=='o' && chr_lc(raw[3])=='m' && chr_lc(raw[4])=='e')
                || (chr_lc(raw[0])=='$' && chr_lc(raw[1])=='{' && chr_lc(raw[2])=='h' && chr_lc(raw[3])=='o' && chr_lc(raw[4])=='m' && chr_lc(raw[5])=='e' && raw[6]=='}') && !raw[7]
                || (chr_lc(raw[0])=='~');
        return home;
    }
    /* exact system dir */
    static const char *sys[]={"/","/root","/etc","/var","/usr","/bin","/sbin","/lib","/lib64","/opt","/home","/boot",NULL};
    for(int i=0;sys[i];i++) if(!strcmp(path,sys[i])) return 1;
    if(star){
        for(int i=0;sys[i];i++){
            if(path[0] && !strcmp(path,sys[i])) return 1;
        }
        /* also direct child of a system dir: /var/log etc. */
        for(int i=0;path[i] && i<520-1;i++){
            if(path[i]=='/'){
                char par[520]; size_t m=0; for(size_t z=0;z<(size_t)i && m+1<sizeof(par);z++) par[m++]=path[z]; par[m]=0;
                for(int s=0;sys[s];s++) if(par[0] && !strcmp(par,sys[s])) return 1;
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* per-pipeline intent predicates                                     */
/* ------------------------------------------------------------------ */
static int intent_pipeline(const char *stmt, int depth){
    if(depth>3) return 0;
    /* ---- stage split on single '|' ---- */
    char st[1240];
    { size_t n=0; while(stmt[n] && n+1<sizeof(st)){ st[n]=stmt[n]; n++; } st[n]=0; }
    char *stage[32]; int ns=0;
    char *save=NULL; char *tk=strtok_r(st,"|",&save);
    while(tk && ns<32){ stage[ns++]=tk; tk=strtok_r(NULL,"|",&save); }
    if(ns<=0) return 0;

    /* download-and-execute across pipeline: curl/wget ... | sh|bash|python... */
    if(has_sub(stmt,"(){ :|:& };:")||has_sub(stmt,":(){ :|:&};:")) return 1;
    int dl=0, ex=0;
    for(int i=0;i<ns;i++){
        if(stage_first_word(stage[i],"curl")||stage_first_word(stage[i],"wget")) dl=1;
        if(stage_first_word(stage[i],"sh")||stage_first_word(stage[i],"bash")
         ||stage_first_word(stage[i],"zsh")||stage_first_word(stage[i],"dash")
         ||stage_first_word(stage[i],"ksh")||stage_first_word(stage[i],"python")
         ||stage_first_word(stage[i],"python3")||stage_first_word(stage[i],"perl")
         ||stage_first_word(stage[i],"ruby")||stage_first_word(stage[i],"php")
         ||stage_first_word(stage[i],"node")) ex=1;
    }
    if(dl&&ex) return 1;
    if((has_sub(stmt,"curl ")||has_sub(stmt,"wget ")) && has_sub(stmt,"<(") && ex) return 1;

    /* base64 pipeline: decode what feeds `base64 -d` and analyze its intent */
    for(int i=0;i<ns;i++){
        if(stage_first_word(stage[i],"base64") && (token_is_in(stage[i],"-d")||token_is_in(stage[i],"--decode"))){
            const char *payload = "";
            if(i>0){
                const char *q=stage[i-1];
                while(*q==' ') q++;
                while(*q && *q!=' ') q++;       /* drop e.g. "echo " */
                while(*q==' ') q++;
                payload=q;
            }
            char dec[560];
            if(payload[0] && base64_decode(payload,dec,sizeof dec) && dec[0]){
                if(intent_statement(dec, depth+1)) return 1;
            }
        }
    }

    /* per-stage semantic predicates */
    for(int i=0;i<ns;i++){
        char *s=stage[i];
        while(*s==' ') s++;
        /* sh -c unwrap: extract next quoted/literal payload arg after -c */
        const char *w1=s; while(*w1 && *w1!=' ') w1++;
        if(( w1-s==2 && chr_lc(s[0])=='s' && chr_lc(s[1])=='h') ||
           ( w1-s==4 && chr_lc(s[0])=='b' && chr_lc(s[1])=='a' && chr_lc(s[2])=='s' && chr_lc(s[3])=='h')){
            char *q=(char*)w1+1;
            while(*q && (*q==' '||*q=='\t')) q++;
            if(has_sub(q,"-c")){
                char payload[520]; size_t k=0;
                const char *r=q;
                /* find argument after -c that is quoted */
                const char *arg=NULL; const char *t=q;
                while(*t){ if(t[0]=='-' && (t[1]=='c') && (t[2]==0||t[2]==' ')){ t+=(t[2]==' ')?3:2; arg=t; break; } t++; }
                while(arg && (*arg==' '||*arg=='\t')) arg++;
                if(arg){ const char *e=arg; if(e[0]=='\''||e[0]=='"'){ char qt=e[0]; e++; while(*e && *e!=qt) e++; }
                    size_t n2=0; const char *z=arg; if(z[0]=='\''||z[0]=='"') z++;
                    while(z<e){ if(n2+1<sizeof(payload)) payload[n2++]=*z; z++; }
                    payload[n2]=0;
                    if(intent_statement(payload, depth+1)) return 1;
                }
            }
        }

        /* destructive delete of dangerous path */
        int destructive = token_is_in(s,"rm") || (token_is_in(s,"busybox")&&token_is_in(s,"rm"))
                        || (token_is_in(s,"toybox")&&token_is_in(s,"rm"))
                        || (token_is_in(s,"find")&&(token_is_in(s,"-delete")||token_is_in(s,"-exec")));
        int forceful = token_is_in(s,"-rf")||token_is_in(s,"-fr")||token_is_in(s,"--recursive")
                     || token_is_in(s,"--force")||(token_is_in(s,"find")&&token_is_in(s,"-delete"));
        if(destructive && forceful){
            const char *p=s; 
            while(*p){
                while(*p==' ') p++;
                const char *b=p; while(*p&&*p!=' ') p++;
                char tok[560]; size_t n=(size_t)(p-b); if(n>=sizeof(tok)) n=sizeof(tok)-1;
                memcpy(tok,b,n); tok[n]=0;
                if(path_danger(tok)) return 1;
            }
        }

        /* raw device write: dd of=/dev/.. or redirection > /dev/.. */
        const char *p=s;
        int redir=0;
        while(*p){
            while(*p==' ') p++;
            const char *b=p; while(*p&&*p!=' ') p++;
            char tok[560]; size_t n=(size_t)(p-b); if(n>=sizeof(tok)) n=sizeof(tok)-1;
            memcpy(tok,b,n); tok[n]=0;
            if(redir && device_path(tok)) return 1;
            redir = (!strcmp(tok,">")||!strcmp(tok,">>")||!strcmp(tok,"1>")||!strcmp(tok,"2>")||!strcmp(tok,"&>"));
            if(chr_lc(tok[0])=='o' && chr_lc(tok[1])=='f' && tok[2]=='=' && device_path(tok+3)) return 1;
            if(token_is_in(s,"tee") && device_path(tok) && tok[0]=='/') return 1;
        }

        /* filesystem creation / wipe on raw device */
        if((token_is_in(s,"mkfs")||has_sub(s,"mkfs")||token_is_in(s,"mke2fs")
           ||token_is_in(s,"wipefs")||token_is_in(s,"fdisk")||token_is_in(s,"parted")
           ||(token_is_in(s,"shred")&&has_sub(s,"/dev/"))||has_sub(s,"debugfs -w"))
           && has_sub(s,"/dev/")) return 1;

        /* uid-0 backdoor */
        if((token_is_in(s,"useradd")||token_is_in(s,"usermod")) && has_sub(s,"-u 0")) return 1;
        /* kernel crash triggers */
        if(has_sub(s,"/proc/sysrq-trigger")||has_sub(s,"sys.kernel.panic")) return 1;
        /* kill pid1 / all */
        if(has_sub(s,"kill -9 1")||has_sub(s,"kill -9 -1")) return 1;
        /* blanket perms on system path */
        if(has_sub(s,"chmod") && (token_is_in(s,"-r")||token_is_in(s,"-R"))
           && (has_sub(s,"000 /")||has_sub(s,"777 /"))) return 1;
        if(has_sub(s,"--no-preserve-root")) return 1;
        /* fork bombs */
        if(has_sub(s,"(){ :|:& };:")||has_sub(s,":(){ :|:&};:")) return 1;
        /* reverse / bind shells */
        if(has_sub(s,"/dev/tcp/")||has_sub(s,"/dev/udp/")) return 1;
        if((has_sub(s,"nc -e ")||has_sub(s,"ncat -e ")||has_sub(s,"nc -c ")||has_sub(s,"ncat --exec"))
           && has_sub(s,"/bin/sh")) return 1;
        if(has_sub(s,"socat") && has_sub(s,"exec:")) return 1;
        if(has_sub(s,"bash -i >&")||has_sub(s,"bash -i &>")) return 1;
    }
    return 0;
}

static int intent_statement(const char *s, int depth){
    if(depth>3) return 0;
    /* fork-bomb signatures span ';', so test the whole statement first */
    if((has_sub(s,"(){") && has_sub(s,"|:&")) ) return 1;
    char st[1200];
    size_t j=0;
    /* split on ; && || and newlines; analyze each segment as a pipeline */
    const char *p=s;
    while(*p && j+1<sizeof(st)){
        if(*p==';' || (*p=='&' && p[1]=='&') || (*p=='|' && p[1]=='|') || *p=='\n'){
            st[j]=0;
            if(j && intent_pipeline(st, depth)) return 1;
            j=0;
            if(*p==';'||*p=='\n') p++;
            else p+=2;
            continue;
        }
        st[j++]=*p++;
    }
    st[j]=0;
    if(j && intent_pipeline(st, depth)) return 1;
    return 0;
}

/* entry: whole prefix line (\x1f-separated commands) */
static int prefix_intent_alert(const char *line){
    const char *p=line;
    while(*p){
        const char *start=p;
        while(*p && *p!='\x1f') p++;
        char cbuf[560]; int cn=0;
        const char *q=start;
        while(q<p && *q!='\t' && cn<(int)sizeof(cbuf)-1) cbuf[cn++]=*q++;
        cbuf[cn]=0;
        char db[1200];
        deobfuscate(cbuf, db, sizeof db);
        if(intent_statement(db, 0)) return 1;
        if(*p) p++;
    }
    return 0;
}

#endif
