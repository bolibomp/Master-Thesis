import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
def Casualties_max(l,certain_Survivor): #takes a list of fitnes values and how many are certain survivors and victims
    Max=[]
    Min=[]
    for i in range(certain_Survivor):
        Max.append(max(l)) #append max of list in a new list
        l.remove(max(l)) #remove max of list from the list
        Min.append(min(l))#append min of list in a new list
        l.remove(min(l))#remove min of list from the list
    for i in range(int(len(l)/2)): #The last consists of elemenst not in the topsheet or bottomsheet and we only want half of what is left        
        random_item_from_list = random.choice(l) #the middle part of the fitness list is choosen at random
        l.remove(random_item_from_list)
    return Max+l

def Casualties_min(l,certain_Survivor): #takes a list of fitnes values and how many are certain survivors and victims
    Max=[]
    Min=[]
    for i in range(certain_Survivor):
        Max.append(max(l)) #append max of list in a new list
        l.remove(max(l)) #remove max of list from the list
        Min.append(min(l))#append min of list in a new list
        l.remove(min(l))#remove min of list from the list
    for i in range(int(len(l)/2)): #The last consists of elemenst not in the topsheet or bottomsheet and we only want half of what is left        
        random_item_from_list = random.choice(l) #the middle part of the fitness list is choosen at random
        l.remove(random_item_from_list)
    return Min+l

def minelement(arr): #<----------------------AxisError: axis 1 is out of bounds for array of dimension 1
    return np.amin(arr, axis=1)

def maxelement(arr): 
    return np.amax(arr, axis=1)

def fit_test_sort(fittness,fittness_test): #takes out the corresponding fittnes from the test data
    A=[]
    B=np.argmin(fittness, axis=1) #find the index, in each row, of the smallest value
    for i,lit in enumerate(fittness_test):#use the index to find the right fittnes in the test fittnes list
        A.append(lit[B[i]])
    return A 
def listevo(lis):
    return np.asarray([lis[i + 1]/lis[i] for i in range(len(lis) - 1)]) 

rand_posmat=[]               
memo_posmat=[]
slump_posmat=[]
def Comparison(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population = 20, survivors = 20, 
               mutation_width = 0.1, width_decay = 0.95, dir_decay = 0, weight_distr = 0.1, hidden_layer = 2, all_plots = False, topology = None):
    def formula(x,surv=0.2):
        return((1)/(x)-1)/(surv)+1
    boost=formula(width_decay)
    class Individual:
        def __init__(self,data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
               mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology):
            self.data_train = data_train
            self.target_train = target_train
            self.data_test = data_test
            self.target_test = target_test
            self.problem = problem
            self.kind = kind
            self.epochs = epochs
            self.iterations = iterations
            self.population = population
            self.survivors = survivors
            self.mutation_width = mutation_width
            self.boost = boost
            self.width_decay = width_decay
            self.dir_decay = dir_decay
            self.weight_distr = weight_distr
            self.hidden_layer = hidden_layer
            self.all_plots = all_plots
            self.topology = topology
            self.positionmatrix = None 
            self.positionvector = None
            self.bias = None
            self.direction = None 
            self.fitness = None
            self.fitness_test = None
            self.counter = 0
            self.marker = 0
            
        #------------------------------------------------------------------------------
                
        def Age_counter(self):
            self.counter = self.counter+1
        #-----------------------------------------------------------------------    
        def Construct_bias(self):
            if self.bias is None:
                bias_size = self.topology[1:]
                bias=[]
                for i in bias_size:
                    bias.append(np.random.normal(loc=0, scale=1, size=i))
                self.bias=np.asarray(bias)
            else:
                pass          
        #------------------------------------------------------------------------------
        
        def Construct_positionmatrix(self):
            if self.positionmatrix is None:
                if self.positionvector is None:
                    posmat = []
                    pairs = [[self.topology[i], self.topology[i + 1]] for i in range(len(self.topology) - 1)]#size of the weight matrix dependce on the topology
                    for current_item, next_item in pairs:
                        posmat.append(np.random.normal(loc = 0, scale=1/np.sqrt(self.data_train.shape[1]), size = (current_item,next_item))) #each weights is drawn from a gaussian 
                        #posmat.append(np.random.uniform(-0.1, 0.1, size=(current_item,next_item)))
                        #with variance depending on the number of inputs
            
                    self.positionmatrix = np.array(posmat)

                else:
                    mat_pos_vec = self.positionvector[:-sum(len(x) for x in self.bias)]
                    bias_pos_vec = self.positionvector[-sum(len(x) for x in self.bias):]
                    self.positionmatrix = self.Vector_to_mat(mat_pos_vec)
                    self.bias = self.Bias_to_mat(bias_pos_vec)
                
            else:
                pass
          
        #------------------------------------------------------------------------------
        
        def Vector_to_mat(self,vector_pop_weights): #take a vector and turning in to several matricies depending on the topology
            pairs = [[self.topology[i], self.topology[i + 1]] for i in range(len(self.topology) - 1)]
            mat_weights = []
            start = 0
            end = 0
            for layer_idx in range(len(self.topology)-1):
                end = end + self.topology[layer_idx] * self.topology[layer_idx + 1]
                curr_vector = vector_pop_weights[start:end]
                mat_layer_weights = np.reshape(curr_vector, newshape = (tuple(pairs[layer_idx])))
                mat_weights.append(mat_layer_weights)
                start = end
            return np.reshape(mat_weights, newshape = (len(self.topology)-1,))
        
        #-----------------------------------------------------------------------
        
        def Bias_to_mat(self,vector_pop_weights):
            lis=vector_pop_weights.copy()
            temp_top = self.topology[1:]
            temp = []
            for i in temp_top:
              temp.append(np.asarray([lis[j] for j in range(i)]))
              lis=lis[i:]
            return(np.asarray(temp))        
        
        #------------------------------------------------------------------------------           
        def Construct_positionvector(self):
             
            if self.positionvector is None:    
                for i in self.positionmatrix:
                    self.positionmatrix.flatten()
                    temp = [i.flatten() for i in self.positionmatrix]
                    pos_temp = np.concatenate(temp, axis=0)
                    
                bias_temp = np.concatenate(self.bias)
                
                self.positionvector = np.concatenate((pos_temp,bias_temp), axis=None)
                    
            else:
                pass 
            
        #----------------------------------------------------------------------
            
        def Mat_to_vec(self,mat):
            mat=np.asarray(mat)
            for i in mat:
                mat.flatten()
                temp = [i.flatten() for i in mat]
                vec_temp = np.concatenate(temp, axis=0)
            return(vec_temp)
        
        #------------------------------------------------------------------------------
                
        def Construct_direction(self):#initialize a random direction
            
            if self.direction is None:
                self.direction = np.zeros(len(self.positionvector))#np.random.normal(0, scale=1/np.sqrt(self.data_train.shape[1]), size=len(self.positionvector))
            else:
                pass        
        
        #------------------------------------------------------------------------------
                
        def Relu(self,x):
            return np.maximum(x, 0, x)
        
        def Sigmoid(self,inpt):
            return 1 / (1 + np.exp(-1 * inpt))
        
        def Softmax(self,inpt):
            expo = np.nan_to_num(np.exp(inpt))
            expo_sum = np.nan_to_num(np.sum(np.exp(inpt)))
            return np.nan_to_num(expo/expo_sum) 
        
        
        def hotone(self,target,kind):
            temp1=[]
            for i in target:
                temp2=[0]*kind
                temp2[i]=1
                temp1.append(temp2)
            return(np.asarray(temp1))
        
        #------------------------------------------------------------------------------
            
        def Error(self): 
            if self.problem == 'Reg': 
                if self.fitness is None:
                    Y = []
                    for sample_idx in range(self.data_train.shape[0]):
                        r1 = self.data_train[sample_idx, :]
                        for idx,curr_weights in enumerate(self.positionmatrix): #run the MLP
                            r1 = np.add(np.matmul(r1, curr_weights),self.bias[idx])
                            if idx == len(self.positionmatrix)-1:
                                if self.kind>2:
                                    r1=r1-np.amax(r1)
                                    r1 = self.Softmax(r1)
                                else:
                                    r1 = self.Sigmoid(r1)
                            else:
                                r1 = self.Relu(r1)
                        if self.kind>2:
                            Y.append(r1)
                        else:
#                            Y.append(np.where(r1 == np.max(r1))[0][0])
                            Y.append(r1)
#                    self.fitness=1/len(self.target_train)*np.sum((np.asarray(Y)-self.target_train)**2)
                    self.fitness=log_loss(self.target_train,Y)+0.0001*1/2*np.sum(self.Mat_to_vec(self.positionmatrix)**2)
                else:
                    pass                                           
            if self.problem == 'Class':
                if self.fitness is None:
                    Y=[]
                    for sample_idx in range(self.data_train.shape[0]): #for each samepl (patient in the cancer case)
                        r1 = self.data_train[sample_idx, :]
                        for idx,curr_weights in enumerate(self.positionmatrix): #run the MLP
                            r1 = np.add(np.matmul(r1, curr_weights),self.bias[idx])
                            if idx == len(self.positionmatrix)-1:
                                if self.kind>2:
                                    r1=r1-np.amax(r1)
                                    r1 = self.Softmax(r1)
                                else:
                                    r1 = self.Sigmoid(r1)
                            else:
                                r1 = self.Relu(r1)
                        if self.kind>2:
                            Y.append(r1)
                        else:
                            Y.append(np.where(r1 == np.max(r1))[0][0])
                    self.fitness=roc_auc_score(y_train, Y)
                else:
                    pass 
            
            #------------------------------------------------------------------------------
           
        def Error_test(self): 
            if self.problem == 'Reg': 
                if self.fitness_test is None:
                    Y = []
                    for sample_idx in range(self.data_test.shape[0]):
                        r1 = self.data_test[sample_idx, :]
                        for idx,curr_weights in enumerate(self.positionmatrix):
                            r1 = np.add(np.matmul(r1, curr_weights),self.bias[idx])
                            if idx == len(self.positionmatrix)-1:
                                if self.kind>2:
                                    r1=r1-np.amax(r1)
                                    r1 = self.Softmax(r1)
                                else:
                                    r1 = self.Sigmoid(r1)
                            else:
                                r1 = self.Relu(r1)
                        if self.kind>2:
                            Y.append(r1)
                        else:
                            #Y.append(np.where(r1 == np.max(r1))[0][0])
                            Y.append(r1)
                    #self.fitness_test=1/len(self.target_test)*np.sum((np.asarray(Y)-self.target_test)**2)
                    self.fitness_test=log_loss(self.target_test,Y)+0.0001*1/2*np.sum(self.Mat_to_vec(self.positionmatrix)**2)
                else:
                    pass
            if self.problem == 'Class':
                if self.fitness_test is None: 
                    Y=[]
                    for sample_idx in range(self.data_test.shape[0]): #for each samepl (patient in the cancer case)
                        r1 = self.data_test[sample_idx, :]
                        for idx,curr_weights in enumerate(self.positionmatrix): #run the MLP
                            r1 = np.add(np.matmul(r1, curr_weights),self.bias[idx])
                            if idx == len(self.positionmatrix)-1:
                                if self.kind>2:
                                    r1=r1-np.amax(r1)
                                    r1 = self.Softmax(r1)
                                else:
                                    r1 = self.Sigmoid(r1)
                            else:
                                r1 = self.Relu(r1)
                        if self.kind>2:
                            Y.append(r1)
                        else:
                            Y.append(np.where(r1 == np.max(r1))[0][0])
                    self.fitness_test=roc_auc_score(y_test, Y)
                else:
                    pass            
            
        #------------------------------------------------------------------------------
            
        def Run(self):        
            self.Age_counter()
            self.Construct_bias()
            self.Construct_positionmatrix()
            self.Construct_positionvector()
            self.Construct_direction()
            self.Error()
            self.Error_test()

            
        #------------------------------------------------------------------------------

#------------------------------------------------------------------------------
    
    total_fitness_Rand = []
    total_fitness_Rand_test = []
    total_numb_new = []
    total_mut = []
    total_epoch = []
    
    
            
    for index in range(iterations):  
        Rand_pop = [Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
                mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology) for _ in range(population)]

        epoch_number=epochs
        epoch_list=[]
        fitness_each_generation_Rand = []
        fitness_each_generation_test_Rand = []
        mutation_each_generation_Rand  = []
        oldest_individual_Rand  = []
        lenght_of_directions_Rand  = []
        number_of_newborn_survivors_Rand  = []
           
        
        for i in range(epochs):
            print(i+1, 'out of', epochs, 'generations, for', index+1, 'iterations out of', iterations)
            
            pop_new_Rand = []
            F_ness_Rand = []
            temp_1_Rand = []
            temp_2_Rand = []
            temp_3_Rand = []
            temp_4_Rand = []
            temp_5_Rand = []
            temp_6_Rand = []
            
            
            for individual in Rand_pop:
                individual.Run()
                individual.marker = 0
                

            for individual in Rand_pop:
                Ind_new = Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
                mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology)
                Ind_new.positionvector = individual.positionvector + individual.mutation_width*np.random.normal(0, 1, len(individual.positionvector))
                Ind_new.mutation_width = individual.mutation_width*boost
                Ind_new.marker = 1
                pop_new_Rand.append(Ind_new) 
                
                
            for Ind_new in pop_new_Rand:
                Ind_new.Run()
                
            for individual in Rand_pop: 
                temp_1_Rand.append(individual.fitness)
                temp_2_Rand.append(individual.mutation_width)
                temp_3_Rand.append(individual.counter)
                temp_4_Rand.append(np.linalg.norm(individual.direction))
                temp_6_Rand.append(individual.fitness_test)
                if individual.fitness == 0:
                    epoch_list.append(i+1)
            if len(epoch_list)>0:
                epoch_number=min(epoch_list)

            fitness_each_generation_Rand.append(temp_1_Rand)
            mutation_each_generation_Rand.append(temp_2_Rand)
            oldest_individual_Rand.append(temp_3_Rand)
            lenght_of_directions_Rand.append(temp_4_Rand)
            fitness_each_generation_test_Rand.append(temp_6_Rand)
            
                 
            Rand_pop = Rand_pop + pop_new_Rand

  
            
            for individual in Rand_pop:    
                F_ness_Rand.append(individual.fitness)
            if Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
                mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).problem == 'Reg':
                casualties_Rand = Casualties_max(F_ness_Rand,survivors)
            if Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
                mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).problem == 'Class':
                casualties_Rand = Casualties_min(F_ness_Rand,survivors)
        
            
            while len(casualties_Rand)>0:
                for idx, individual in enumerate(Rand_pop):
                    if individual.fitness in casualties_Rand:
                        del Rand_pop[idx]
                        casualties_Rand.remove(individual.fitness)
                        
                        
            for individual in Rand_pop:
                individual.mutation_width =individual.mutation_width*width_decay
                if individual.marker == 1: 
                    temp_5_Rand.append(individual.marker)
            
            number_of_newborn_survivors_Rand.append(temp_5_Rand)
            
            # if len(number_of_newborn_survivors_Rand[i])/population>1/5:
            #     for individual in Rand_pop:
            #         individual.mutation_width =individual.mutation_width/0.817
            # if len(number_of_newborn_survivors_Rand[i])/population<1/5:
            #     for individual in Rand_pop:
            #         individual.mutation_width =individual.mutation_width*0.817 
           

        Averge_mutation_width_Rand=[sum(mutation_each_generation_Rand[i])/population for i in range(epochs)]

        total_mut.append(Averge_mutation_width_Rand)


        Number_of_Newborn_Survivors_Rand=[len(number_of_newborn_survivors_Rand[i]) for i in range(epochs)]
        total_numb_new.append(Number_of_Newborn_Survivors_Rand)
        

        total_epoch.append(epoch_number)
        if Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
               mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).problem == 'Reg':
             total_fitness_Rand.append(minelement(fitness_each_generation_Rand))
             total_fitness_Rand_test.append(fit_test_sort(fitness_each_generation_Rand,fitness_each_generation_test_Rand))
        if Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
               mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).problem == 'Class':
            total_fitness_Rand.append(maxelement(fitness_each_generation_Rand))
            total_fitness_Rand_test.append(fit_test_sort(fitness_each_generation_Rand,fitness_each_generation_test_Rand))
             
    population=Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
               mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).population
    boost=Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
               mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).boost
    decay=Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
               mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).width_decay            
    survivors=Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
                mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).survivors
    epochs=Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, iterations, population, survivors, 
                mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).epochs
    mut_width=Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, 
                     iterations, population, survivors, mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).mutation_width
    dir_decay=Individual(data_train, target_train, data_test, target_test, problem, kind, epochs, 
                     iterations, population, survivors, mutation_width, boost, width_decay, dir_decay, weight_distr, hidden_layer, all_plots,topology).dir_decay

    average_fit=1/len(total_fitness_Rand)*np.sum(total_fitness_Rand,axis=0)
    average_newB=1/len(total_numb_new)*np.sum(np.array(total_numb_new)/population,axis=0)
    average_mut=1/(len(total_mut))*np.sum(total_mut,axis=0)
    std_fit=np.array(total_fitness_Rand).std(0)
    return(average_fit,average_newB,average_mut,population,boost,decay,survivors,mut_width,dir_decay,std_fit)

    
def spirals(n_points, noise=0, nturn=1):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * nturn * (2*np.pi)
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))


nturn=1.5
x,y = spirals(1000, 0, nturn)

x[:,0]=(x[:,0]-np.mean(x[:,0]))/(np.std(x[:,0]))
x[:,1]=(x[:,1]-np.mean(x[:,1]))/(np.std(x[:,1]))

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)


start_time = time.clock()

results=Comparison(x_train, y_train, x_test, y_test,'Reg', 1, 300, 25 , 
topology=[2,20,20,1],population=50,survivors=40,width_decay=0.99,mutation_width=0.1,dir_decay=0)


time = (time.clock() - start_time)*0.0166666667

if time < 1:
    print('The code took', round(time/0.0166666667,2), 'seconds to complete')
else:
    temp_1=int(time)
    temp_2=time-temp_1
    print('The code took', temp_1, 'minutes and', round(temp_2/0.0166666667,2),'seconds to complete')

f = open('15DW.txt','w')
for i in range(len(results)):
    if isinstance(results[i], np.ndarray):
        f.write(str(list(results[i])))
        f.write('\n')
    if isinstance(results[i], float) or isinstance(results[i], int):
        f.write(str(results[i]))
        f.write('\n')
f.close()
