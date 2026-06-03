a = [[[[1,2,3,4]]],[5,6],[[1,2]]]
lst=[]
for i in a[0][0][0]:
    # print(i)
    lst.append(i)
    # for j in i:
        # print(j)
for i in a[1]:
    # print(i)  
    lst.append(i)

for i in a[2]:
    (lst.extend(i))
print(lst)
