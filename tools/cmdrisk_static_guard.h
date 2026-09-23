/* Static critical-pattern guard for catastrophic one-liner commands.
 *
 * Background: the calibrated model is trained on multi-command session
 * prefixes with time deltas.  A bare single command (out-of-distribution)
 * can score catastrophic destroy/backdoor/lateral one-liners below the
 * session-FPR 1% threshold even though the semantic risk is extreme
 * (example: "rm -rf /" scores 3.36e-05 < 1.18e-04 on v5-L8).
 *
 * These checks are deterministic and conservative:
 *   - they only FORCE decision=1 (alert), never decision=0;
 *   - they are alert-only semantics.  Human review is still required and
 *     the tool never blocks anything.
 *
 * A fifth output column reports rule_hit (1/0).  See
 * reports/STATIC_GUARD_v1.1.2_zh.md and ci_assets/regression_critical.txt.
 */
#ifndef CMDRISK_STATIC_GUARD_H
#define CMDRISK_STATIC_GUARD_H

#include <string.h>

static int chr_lc(char a){ return (a>='A' && a<='Z') ? (a-'A'+'a') : a; }

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

/* lowercased copy; returns length or -1 if truncated */
static int lower_copy(const char *src, char *dst, size_t cap){
    size_t i=0;
    for (; *src && i+1<cap; src++, i++) dst[i]=(char)chr_lc(*src);
    dst[i]=0;
    return (int)i;
}

static int critical_cmd(const char *cmd){
    char c[512];
    lower_copy(cmd, c, sizeof c);

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

    /* rm -rf on whole-system / whole-home targets: exact-argument scan */
    if (has_sub(c,"rm ")){
        char t[512];
        strcpy(t, c);
        char *tok = strtok(t, " \t\r\n");
        int seen_rf = 0;
        while (tok){
            if (!strcmp(tok,"-rf") || !strcmp(tok,"-fr")
             || !strcmp(tok,"--recursive") || !strcmp(tok,"--force")){
                seen_rf = 1;
            } else if (tok[0] != '-'){
                int system_target =
                    !strcmp(tok,"/")   || !strcmp(tok,"/*")  ||
                    !strcmp(tok,"/root")||!strcmp(tok,"/etc")||
                    !strcmp(tok,"/var")||!strcmp(tok,"/usr")||
                    !strcmp(tok,"/bin")||!strcmp(tok,"/sbin")||
                    !strcmp(tok,"/lib")||!strcmp(tok,"/lib64")||
                    !strcmp(tok,"/opt")||!strcmp(tok,"/home")||
                    !strcmp(tok,"/boot")||!strcmp(tok,"$home")||
                    !strcmp(tok,"~")   || !strcmp(tok,"/etc/")||
                    !strcmp(tok,"/var/");
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
