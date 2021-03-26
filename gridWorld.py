def Gridworld():
    
    mat = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    R = -1
    p = 0.25
    Vk = 0
    row_idx = list(range(len(mat)))
    col_idx = len(mat[0])
    
    final_mat = []
    for a in range(n):
        mid_mat = []
        for r in row_idx:
            first_mat = []
            for c in range(col_idx):
                OM_r = r-1
                if r == 0:
                    OM_r += 1
                    
                OP_r= r+1
                if r == 3:
                    OP_r -= 1
                    
                OM_c = c-1
                if c == 0:
                    OM_c += 1
                    
                OP_c = c+1
                if c == 3:
                    OP_c -= 1
                
                Vk = R + p*(mat[OM_r][c]+mat[OP_r][c]+mat[r][OP_c]+mat[r][OM_c])

                first_mat.append(Vk)
    
            mid_mat.append(first_mat)            

        
        mid_mat[0][0] = 0
        mid_mat[3][3] = 0

        mat=[]
        mat=mid_mat

    final_mat.append(mid_mat)

    return final_mat
        
n = int(input("k를 몇회 하실건가요?\n"))

from pprint import pprint
pprint(Gridworld())