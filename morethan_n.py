def morethan_n(input_file, n):
    max_len = 0
    f = open(input_file,"r") 
    i = 0
    while True: 
        i += 1
        line = f.readline() 
        line = line[2:-1] #去掉换行符
        if not line:
            break
        length = len(line.split(' '))
        if length > n:
            max_len += 1
    return max_len
print(morethan_n("./datasets/train.txt", 50))
