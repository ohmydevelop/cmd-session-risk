/* Static critical-pattern guard for catastrophic one-liner commands.
 *
 * Positioning (important): this layer is NOT the detector and is not meant to
 * "generalize".  The BERT model generalizes; this layer CERTIFIES a small set
 * of high-blast-radius, semantically exhaustive predicates that a corpus model
 * cannot learn reliably from benign data alone ("rm -rf /" vs "rm -rf /tmp/a"
 * differ by one token, but benign data never contains a bare "/").
 *
 * It only FORCES decision=1 (alert, human review); it never suppresses the
 * model and never blocks anything.  Deterministic, auditable, zero-FN for the
 * canonical spellings, and continuously widened by a de-obfuscation
 * normalizer in front of the matcher (see v1.1.3 notes).
 */
#ifndef CMDRISK_STATIC_GUARD_H
#define CMDRISK_STATIC_GUARD_H

#include <string.h>

static int chr_lc(char a){ return (a>='A' && a<='Z') ? (a-'A'+'a') : a; }
static int is_hexc(char c){ return (c>='0'&&c<='9')||(c>='a'&&c<='f')||(c>='A'&&c<='F'); }
static int hex_val(char c){
    if(c>='0'&&c<='9') return c-'0';
    c = chr_lc(c);
    return (c>='a' && c<='f') ? (c-'a'+10) : 0;
}

static int has_sub(const char *hay, const char *needle){
    if (!needle[0]) return 1;
    for (; *hay; hay++){
        const char *h=hay, *n=needle;
        while (*h && *n){
            if (chr_lc(*h) != chr_lc(*n)) break;
            h++; n++;
        }
        if (!*n) return 1;
    }
    return 0;
}

static int lower_copy(const char *src, char *dst, size_t cap){
    size_t i=0;
    for (; *src && i+1<cap; src++, i++) dst[i]=(char)chr_lc(*src);
    dst[i]=0;
    return (int)i;
}

/* ------------------------------------------------------------------ */
/* De-obfuscation normalizer for the guard layer (deterministic)      */
/*   - drop shell quotes so r'm' / "r"m collapse to rm                */
/*   - unescape backslash sequences: \xNN -> byte, \<c> -> c          */
/*   - resolve trivial $(echo X) / `echo X` substitutions; anything   */
/*     else becomes a placeholder token                               */
/*   - collapse whitespace, "//" and "/./" in path tokens             */
/* ------------------------------------------------------------------ */
static void deobfuscate(const char *src, char *dst, size_t cap){
    size_t j=0;
    const char *p=src;
    while(*p && j+1<cap){
        char c=*p;
        if(c=='\\'){
            if(p[1]=='x' && is_hexc(p[2]) && is_hexc(p[3])){
                dst[j++]=(char)(hex_val(p[2])*16 + hex_val(p[3])); p+=4; continue;
            }
            if(p[1]){ dst[j++]=(char)p[1]; p+=2; continue; }
            p++; continue;
        }
        if(c=='\'' || c=='"'){ p++; continue; }
        if(c=='$' && p[1]=='('){
            const char *q=p+2; int depth=0;
            while(*q && !(*q==')' && depth==0)){
                if(*q=='(') depth++;
                if(*q==')') depth--;
                q++;
            }
            const char *e=p+2;
            while(e<q && (*e==' '||*e=='\t')) e++;
            if(e+5<=q && chr_lc(e[0])=='e' && chr_lc(e[1])=='c' && chr_lc(e[2])=='h'
               && chr_lc(e[3])=='o' && (e[4]==' '||e[4]=='\t')){
                while(e<q && (*e==' '||*e=='\t')) e++;
                while(e<q && j+1<cap){ dst[j++]=*e; if(*e==0)break; e++; }
            } else if(j+1<cap){ dst[j++]='?'; }
            if(*q==')') q++;
            p=q; continue;
        }
        if(c=='`'){
            const char *q=p+1; while(*q && *q!='`') q++;
            const char *e=p+1;
            while(e<q && (*e==' '||*e=='\t')) e++;
            if(e+5<=q && chr_lc(e[0])=='e' && chr_lc(e[1])=='c' && chr_lc(e[2])=='h'
               && chr_lc(e[3])=='o' && (e[4]==' '||e[4]=='\t')){
                while(e<q && (*e==' '||*e=='\t')) e++;
                while(e<q && j+1<cap){ dst[j++]=*e; if(*e==0)break; e++; }
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
    /* path normalization: collapse "//" runs and drop "/./" */
    char out[520]; size_t k=0; const char *s=dst;
    while(*s && k+1<sizeof(out)){
        if(*s=='/'){
            out[k++]='/'; s++;
            while(*s=='/') s++;
            while(s[0]=='.' && s[1]=='/') s+=2;
            if(s[0]=='.' && !s[1]){ s++; continue; }
            continue;
        }
        out[k++]=*s++;
    }
    out[k]=0;
    if(k+1<=cap){ size_t i=0; for(;out[i];i++) dst[i]=out[i]; dst[i]=0; }
}

/* exact whitespace-token match on the (lowercased, normalized) command */
static int has_exact_arg(const char *cmd, const char *arg){
    const char *p=cmd;
    while(*p){
        while(*p==' ') p++;
        const char *s=p;
        while(*p && *p!=' ') p++;
        if((size_t)(p-s)==strlen(arg) && strncmp(s,arg,(size_t)(p-s))==0) return 1;
        if(!*p) break;
    }
    return 0;
}

static int critical_cmd(const char *cmd){
    char c[512], d[512];
    deobfuscate(cmd, c, sizeof c);
    lower_copy(c, d, sizeof d);
    strcpy(c, d);

    /* root-filesystem wipe */
    if (has_sub(c,"--no-preserve-root")) return 1;
    /* kernel panic / emergency sync triggers */
    if (has_sub(c,"/proc/sysrq-trigger")) return 1;
    if (has_sub(c,"sys.kernel.panic")) return 1;
    /* raw disk / filesystem destruction */
    if (has_sub(c,"mkfs") && has_sub(c,"/dev/")) return 1;
    if (has_sub(c,"dd ") && has_sub(c,"of=/dev/")) return 1;
    if (has_sub(c,"shred ") && has_sub(c,"/dev/")) return 1;
    if ((has_sub(c,"wipefs")||has_sub(c,"fdisk")||has_sub(c,"parted")
        ||has_sub(c,"debugfs -w")) && has_sub(c,"/dev/")) return 1;
    /* uid-0 account creation / escalation */
    if ((has_sub(c,"useradd")||has_sub(c,"usermod")) && has_sub(c,"-u 0")) return 1;
    /* killing pid 1 or every process */
    if (has_sub(c,"kill -9 1") || has_sub(c,"kill -9 -1")) return 1;
    /* blanket recursive permission change on system paths */
    if (has_sub(c,"chmod") && (has_sub(c,"-r ")||has_sub(c,"-r\t"))
        && (has_sub(c,"000 /")||has_sub(c,"777 /"))) return 1;
    /* download-and-execute */
    if ((has_sub(c,"curl ")||has_sub(c,"wget ")) &&
        (has_sub(c,"| sh")||has_sub(c,"|sh")||has_sub(c,"| bash")||has_sub(c,"|bash")||
         has_sub(c,"bash <("))) return 1;
    /* reverse / bind shells */
    if (has_sub(c,"/dev/tcp/")) return 1;
    if ((has_sub(c,"nc -e ")||has_sub(c,"ncat -e ")||has_sub(c,"socat")) && has_sub(c,"/bin/sh")) return 1;
    if (has_sub(c,"bash -i >&")) return 1;
    /* fork-bomb signatures */
    if (has_sub(c,"(){ :|:& };:")) return 1;
    if (has_sub(c,":(){ :|:&};:")) return 1;
    /* find-style recursive deletion of root */
    if (has_sub(c,"find ") && (has_sub(c,"-delete")||has_sub(c,"-exec")) && has_exact_arg(c,"/")) return 1;

    /* recursive force removal of whole-system targets */
    if (has_sub(c,"rm ")){
        char t[512]; strcpy(t, c);
        char *tok = strtok(t, " \t\r\n");
        int seen_rf = 0;
        while (tok){
            if (!strcmp(tok,"-rf") || !strcmp(tok,"-fr") || !strcmp(tok,"--force")
             || !strcmp(tok,"--recursive")) seen_rf = 1;
            else if (tok[0] != '-'){
                int system_target =
                    !strcmp(tok,"/")   || !strcmp(tok,"/*")  ||
                    !strcmp(tok,"/root")||!strcmp(tok,"/etc")||
                    !strcmp(tok,"/var")||!strcmp(tok,"/usr")||
                    !strcmp(tok,"/bin")||!strcmp(tok,"/sbin")||
                    !strcmp(tok,"/lib")||!strcmp(tok,"/lib64")||
                    !strcmp(tok,"/opt")||!strcmp(tok,"/home")||
                    !strcmp(tok,"/boot")||!strcmp(tok,"$home")||
                    !strcmp(tok,"${home}")||!strcmp(tok,"~");
                if (system_target && seen_rf) return 1;
            }
            tok = strtok(NULL, " \t\r\n");
        }
    }
    return 0;
}

/* any command inside a \x1f-separated prefix hits the guard */
static int prefix_critical(const char *line){
    const char *p = line;
    while (*p){
        const char *start = p;
        while (*p && *p != '\x1f') p++;
        char cbuf[512]; int cn = 0;
        const char *q = start;
        while (q < p && *q != '\t' && cn < (int)sizeof(cbuf)-1) cbuf[cn++] = *q++;
        cbuf[cn] = 0;
        if (critical_cmd(cbuf)) return 1;
        if (*p) p++;
    }
    return 0;
}

#endif
