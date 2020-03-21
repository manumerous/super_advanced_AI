# _________                              ____   ____      .__  .__    .___       __  .__               
# \_   ___ \_______  ____  ______ ______ \   \ /   /____  |  | |__| __| _/____ _/  |_|__| ____   ____  
# /    \  \/\_  __ \/  _ \/  ___//  ___/  \   Y   /\__  \ |  | |  |/ __ |\__  \\   __\  |/  _ \ /    \ 
# \     \____|  | \(  <_> )___ \ \___ \    \     /  / __ \|  |_|  / /_/ | / __ \|  | |  (  <_> )   |  \
#  \______  /|__|   \____/____  >____  >    \___/  (____  /____/__\____ |(____  /__| |__|\____/|___|  /
#         \/                  \/     \/                 \/             \/     \/                    \/ 


import numpy as np 
import math

#generates cross folding datasets
def generate_cv_datasets(k, test_set_index, data):
    row_count, _ = data.shape
    rows_per_set = np.zeros(k)
    rows_per_set.fill(math.ceil(row_count/k)-1)
    for i in range(row_count%k):
        rows_per_set[i] += 1
    row_index = np.zeros(k+1).astype(int)
    for i in range(k):
        if (i==0):
            row_index[i]= 0
        else:
            row_index[i]= row_index[i-1] + rows_per_set[i-1]
    row_index[k]=row_count 
    training_set = None
    testing_set = None
    for i in range(k):
        if (i== test_set_index):
            testing_set = data[row_index[i]:(row_index[i+1]), :]
        else:
            if (training_set is None):
                training_set = data[row_index[i]:(row_index[i+1]), :]
            else:
                data_to_append = data[row_index[i]:(row_index[i+1]), :]
                training_set = np.append(training_set, data_to_append, axis=0)
    return testing_set, training_set

def main():

    data = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11]])
    test_set, train_set = generate_cv_datasets(3, 1, data)
    print(test_set)
    print(train_set)


# for isolated testing purposes only:
if __name__ == "__main__":
    main()




    

