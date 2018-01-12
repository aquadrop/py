"""
-------------------------------------------------
   File Name：     regex_rank
   Description :
   Author :       deep
   date：          18-1-12
-------------------------------------------------
   Change Activity:
                   18-1-12:
                   
   __author__ = 'deep'
-------------------------------------------------
"""

import re

def ld(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]

s='Mary had a little lamb'
d={}
regs=[r'.*', r'Mary', r'lamb', r'little lamb', r'.*little lamb',r'\b\w+mb',
        r'Mary.*little lamb',r'.*[lL]ittle [Ll]amb',r'\blittle\b',s,r'little']

def rank_regs(s, reg):
    m = re.search(reg, s)
    if m:
        ld1 = ld(reg, m.group(0))
        ld2 = ld(m.group(0), s)
        score = max(ld1, ld2)
        return 1.0 / (score + 1)
    else:
        return 0.0




if __name__ == '__main__':
    for reg in regs:
        m = re.search(reg, s)
        if m:
            print("'%s' matches '%s' with sub group '%s'" % (reg, s, m.group(0)))
            ld1 = ld(reg, m.group(0))
            ld2 = ld(m.group(0), s)
            score = max(ld1, ld2)
            print("  %i edits regex->match(0), %i edits match(0)->s" % (ld1, ld2))
            print("  score: ", score)
            d[reg] = rank_regs(s=s, reg=reg)
            print
        else:
            print("'%s' does not match '%s'" % (reg, s))

    print("   ===== %s =====    === %s ===" % ('RegEx'.center(10),'Score'.center(10)))

    for key, value in d.items():
        print("   %22s        %5s" % (key, value))

    inputs_interpreter_regex_helper = {'register': r'register|注册',
                                       'ok': r'ok',
                                       'oops': r'不行',
                                       'auth': r'auth'}

    query = 'regi'
    for a, b in inputs_interpreter_regex_helper.items():
        score = rank_regs(query, b)
        print(a, score)


