shell_i = 10
shell_j = 2
shell_k = 6
occup = [str(i) for i in list(range(shell_i+shell_j+shell_k))]
for i in range(shell_i):
    for j in range(shell_i, shell_i+shell_j):
        for k in range(shell_i+shell_j, shell_i+shell_j+shell_k):
            this = occup.copy()
            this.pop(k)
            this.pop(j)
            this.pop(i)
            print(' '.join(this))
            
