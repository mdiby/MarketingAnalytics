# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:08:57 2017

@author: https://jessesw.com/Rec-System/
"""

import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve


path = 'E:\Datasets\Recommender\Online Retail\Online Retail.xlsx'
retail_data = pd.read_excel(path)


cleaned_retail = retail_data.loc[pd.isnull(retail_data.CustomerID) == False]

# Only get unique item/description pairs
item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates() 

# Encode as strings for future lookup ease
item_lookup['StockCode'] = item_lookup.StockCode.astype(str) 


# Convert to int for customer ID
cleaned_retail['CustomerID'] = cleaned_retail.CustomerID.astype(int) 


# Get rid of unnecessary info
cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']] 

grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index() # Group together
# Replace a sum of zero purchases with a one to indicate purchased
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 

# Only get customers where purchase totals were positive
grouped_purchased = grouped_cleaned.query('Quantity > 0') 




# Get our unique customers
customers = list(np.sort(grouped_purchased.CustomerID.unique()))

# Get our unique products that were purchased 
products = list(grouped_purchased.StockCode.unique()) 

# All of our purchases
quantity = list(grouped_purchased.Quantity) 

# Get the associated row indices
rows = grouped_purchased.CustomerID.astype('category', 
                                           categories = customers).cat.codes 
# Get the associated column indices
cols = grouped_purchased.StockCode.astype('category', 
                                          categories = products).cat.codes 

purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), 
                                     shape=(len(customers), len(products)))


# Number of possible interactions in the matrix
matrix_size = purchases_sparse.shape[0]*purchases_sparse.shape[1] 

# Number of items interacted with
num_purchases = len(purchases_sparse.nonzero()[0]) 

sparsity = 100*(1 - (num_purchases/matrix_size))
sparsity



import random




def make_train(ratings, pct_test = 0.2):
    '''
    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings, 
    while the training set replaces the specified percentage of them with a zero in the original ratings matrix. 
    
    parameters: 
    
    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix. 
    
    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the 
    training set for later comparison to the test set, which contains all of the original ratings. 
    
    returns:
    
    training_set - The altered version of the original data with a certain percentage of the user-item pairs 
    that originally had interaction set back to zero.
    
    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order 
    compares with the actual interactions.
    
    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.
    '''
     # Make a copy of the original set to be the test set. 
    test_set = ratings.copy()
    # Store the test set as a binary preference matrix
    test_set[test_set != 0] = 1 
    # Make a copy of the original data we can alter as our training set. 
    training_set = ratings.copy() 
    # Find the indices in the ratings data where an interaction exists
    nonzero_inds = training_set.nonzero() 
    # Zip these pairs together of user,item index into list
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) 
    # Set the random seed to zero for reproducibility
    random.seed(0)
    # Round the number of samples needed to the nearest integer
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) 
    # Sample a random number of user-item pairs without replacement
    samples = random.sample(nonzero_pairs, num_samples) 
    # Get the user row indices
    user_inds = [index[0] for index in samples] 
    # Get the item column indices
    item_inds = [index[1] for index in samples] 
    # Assign all of the randomly chosen user-item pairs to zero
    training_set[user_inds, item_inds] = 0 
    # Get rid of zeros in sparse array storage after update to save space
    training_set.eliminate_zeros() 
    # Output the unique list of user rows that were altered  
    return training_set, test_set, list(set(user_inds)) 





def implicit_weighted_ALS(training_set, 
                          lambda_val = 0.1, 
                          alpha = 40, 
                          iterations = 10, 
                          rank_size = 20, 
                          seed = 0):
    '''
    Implicit weighted ALS taken from Hu, Koren, and Volinsky 2008. 
    Designed for alternating least squares and implicit
    feedback based collaborative filtering. 
    
    parameters:
    
    training_set - Our matrix of ratings with shape m x n, 
    where m is the number of users and n is the number of items.
    Should be a sparse csr matrix to save space. 
    
    lambda_val - Used for regularization during alternating least squares. 
    Increasing this value may increase bias
    but decrease variance. Default is 0.1. 
    
    alpha - The parameter associated with the confidence matrix discussed 
    in the paper, where Cui = 1 + alpha*Rui. 
    The paper found a default of 40 most effective. 
    Decreasing this will decrease the variability in confidence between
    various ratings.
    
    iterations - The number of times to alternate between both user 
    feature vector and item feature vector in
    alternating least squares. More iterations will allow 
    better convergence at the cost of increased computation. 
    The authors found 10 iterations was sufficient, 
    but more may be required to converge. 
    
    rank_size - The number of latent features in the user/item feature vectors. 
    The paper recommends varying this 
    between 20-200. Increasing the number of features 
    may overfit but could reduce bias. 
    
    seed - Set the seed for reproducible results
    
    returns:
    
    The feature vectors for users and items. 
    The dot product of these feature vectors should give you the expected 
    "rating" at each point in your original matrix. 
    '''
    
    # first set up our confidence matrix
    # To allow the matrix to stay sparse, I will add one later when each row is taken 
    # and converted to dense. 
    conf = (alpha*training_set) 
    num_user = conf.shape[0]
    # Get the size of our original ratings matrix, m x n
    num_item = conf.shape[1] 
    
    # initialize our X/Y feature vectors randomly with a set seed
    rstate = np.random.RandomState(seed)
    
    # Random numbers in a m x rank shape
    X = sparse.csr_matrix(rstate.normal(size = (num_user, rank_size))) 
    # Normally this would be rank x n but we can 
    # transpose at the end. Makes calculation more simple
    Y = sparse.csr_matrix(rstate.normal(size = (num_item, rank_size))) 
    
    X_eye = sparse.eye(num_user)
    Y_eye = sparse.eye(num_item)
    
    # Our regularization term lambda*I. 
    lambda_eye = lambda_val * sparse.eye(rank_size)
    
    # We can compute this before iteration starts. 
    # Begin iterations
    # Iterate back and forth between solving X given fixed Y and vice versa
    
    for iter_step in range(iterations): 
        # Compute yTy and xTx at beginning of each iteration to save computing time
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)
        # Being iteration to solve for X based on fixed Y
        for u in range(num_user):
            # Grab user row from confidence matrix and convert to dense
            conf_samp = conf[u,:].toarray() 
            pref = conf_samp.copy() 
            
            # Create binarized preference vector
            pref[pref != 0] = 1  
            # Get Cu - I term, don't need to subtract 1 since we never added it 
            CuI = sparse.diags(conf_samp, [0]) 
            # This is the yT(Cu-I)Y term 
            yTCuIY = Y.T.dot(CuI).dot(Y)
            # This is the yTCuPu term, where we add the eye back in Cu - I + I = Cu
            yTCupu = Y.T.dot(CuI + Y_eye).dot(pref.T) 
            # Solve for Xu = ((yTy + yT(Cu-I)Y + lambda*I)^-1)yTCuPu, 
            # equation 4 from the paper  
            X[u] = spsolve(yTy + yTCuIY + lambda_eye, yTCupu) 
            
        # Begin iteration to solve for Y based on fixed X 
        for i in range(num_item):
            # transpose to get it in row format and convert to dense
            conf_samp = conf[:,i].T.toarray() 
            pref = conf_samp.copy()
            # Create binarized preference vector
            pref[pref != 0] = 1
            # Get Ci - I term, don't need to subtract 1 since we never added it
            CiI = sparse.diags(conf_samp, [0]) 
            # This is the xT(Cu-I)X term
            xTCiIX = X.T.dot(CiI).dot(X) 
            # This is the xTCiPi term
            xTCiPi = X.T.dot(CiI + X_eye).dot(pref.T) 
            # Solve for Yi = ((xTx + xT(Cu-I)X) + lambda*I)^-1)xTCiPi, 
            # equation 5 from the paper
            Y[i] = spsolve(xTx + xTCiIX + lambda_eye, xTCiPi)
            
    # End iterations
    # Transpose at the end to make up for not being transposed at the beginning. 
    # Y needs to be rank x n. Keep these as separate matrices for scale reasons. 
    return X, Y.T 




product_train, product_test, product_users_altered = make_train(purchases_sparse, 
                                                                pct_test = 0.2)


user_vecs, item_vecs = implicit_weighted_ALS(product_train, 
                                             lambda_val = 0.1, 
                                             alpha = 15, 
                                             iterations = 10,
                                             rank_size = 20)


ratings_for_particula_user = user_vecs[0,:].dot(item_vecs).toarray()[0,:5]




from sklearn import metrics


def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics. 
    
    parameters:
    
    - predictions: your prediction output
    
    - test: the actual target result you are comparing to
    
    returns:
    
    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)  



def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user 
    that had their user-item matrix altered. 
    
    parameters:
    
    training_set - The training set resulting from make_train, 
    where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model 
    
    predictions - The matrix of your predicted ratings 
    for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero 
    and item vectors as item one. 
    
    altered_users - The indices of the users where at least one 
    user/item pair was altered from make_train function
    
    test_set - The test set constucted earlier from make_train function
    
    
    
    returns:
    
    The mean AUC (area under the Receiver Operator Characteristic curve) 
    of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition 
    to the most popular items as a benchmark.
    '''
    
    # An empty list to store the AUC for each user that had an item 
    # removed from the training set
    store_auc = [] 
    # To store popular AUC scores
    popularity_auc = [] 
    # Get sum of item iteractions to find most popular
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) 
    item_vecs = predictions[1]
    
    # Iterate through each user that had an item altered
    for user in altered_users: 
        # Get the training set row
        training_row = training_set[user,:].toarray().reshape(-1) 
        # Find where the interaction had not yet occurred
        zero_inds = np.where(training_row == 0) 
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        #print(user_vec.shape,item_vecs.shape)
        pred = user_vec.dot(item_vecs.T).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training 
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration
    
    # Return the mean AUC rounded to three decimal places 
    # for both test and popularity benchmark
    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))  



# AUC for our recommender system
calc_mean_auc(product_train, product_users_altered, 
              [sparse.csr_matrix(user_vecs), 
               sparse.csr_matrix(item_vecs.T)], 
              product_test)



customers_arr = np.array(customers) # Array of customer IDs from the ratings matrix
products_arr = np.array(products) # Array of product IDs from the ratings matrix


def get_items_purchased(customer_id, mf_train, customers_list, 
                        products_list, item_lookup):
    '''
    This just tells me which items have been already purchased 
    by a specific user in the training set. 
    
    parameters: 
    
    customer_id - Input the customer's id number that you want 
    to see prior purchases of at least once
    
    mf_train - The initial ratings training set used (without weights applied)
    
    customers_list - The array of customers used in the ratings matrix
    
    products_list - The array of products used in the ratings matrix
    
    item_lookup - A simple pandas dataframe of the unique product 
    ID/product descriptions available
    
    returns:
    
    A list of item IDs and item descriptions for a particular customer 
    that were already purchased in the training set
    '''
    # Returns the index row of our customer id
    cust_ind = np.where(customers_list == customer_id)[0][0] 
    # Get column indices of purchased items
    purchased_ind = mf_train[cust_ind,:].nonzero()[1] 
    # Get the stock codes for our purchased items
    prod_codes = products_list[purchased_ind] 
    return item_lookup.loc[item_lookup.StockCode.isin(prod_codes)]



customers_arr[:5]

from sklearn.preprocessing import MinMaxScaler



def rec_items(customer_id, mf_train, user_vecs, item_vecs, 
              customer_list, item_list, item_lookup, num_items = 10):
    '''
    This function will return the top recommended items to our users 
    
    parameters:
    
    customer_id - Input the customer's id number 
                  that you want to get recommendations for
    
    mf_train - The training matrix you used for matrix factorization fitting
    
    user_vecs - the user vectors from your fitted matrix factorization
    
    item_vecs - the item vectors from your fitted matrix factorization
    
    customer_list - an array of the customer's ID numbers that make up 
                    the rows of your ratings matrix 
                    (in order of matrix)
    
    item_list - an array of the products that make up the columns of your ratings matrix
                    (in order of matrix)
    
    item_lookup - A simple pandas dataframe of the unique product 
                  ID/product descriptions available
    
    num_items - The number of items you want to recommend 
                in order of best recommendations. Default is 10. 
    
    returns:
    
    -   The top n recommendations chosen based on the user/item vectors 
        for items never interacted with/purchased
    '''
    
    # Returns the index row of our customer id
    cust_ind = np.where(customer_list == customer_id)[0][0] 
    # Get the ratings from the training set ratings matrix
    pref_vec = mf_train[cust_ind,:].toarray() 
    # Add 1 to everything, so that items not purchased yet become equal to 1
    pref_vec = pref_vec.reshape(-1) + 1 
    # Make everything already purchased zero
    pref_vec[pref_vec > 1] = 0 
    # Get dot product of user vector and all item vectors
    rec_vector = user_vecs[cust_ind,:].dot(item_vecs) 
    # Scale this recommendation vector between 0 and 1
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.todense().reshape(-1,1))[:,0] 
    recommend_vector = pref_vec*rec_vector_scaled 
    # Items already purchased have their recommendation multiplied by zero
    # Sort the indices of the items into order # of best recommendations 
    product_idx = np.argsort(recommend_vector)[::-1][:num_items] 
    # start empty list to store items
    rec_list = [] 
    for index in product_idx:
        code = item_list[index]
        # Append our descriptions to the list
        rec_list.append([code, 
                         item_lookup.Description.loc[item_lookup.StockCode == code].iloc[0]]) 
        
    codes = [item[0] for item in rec_list]
    descriptions = [item[1] for item in rec_list]
    # Create a dataframe 
    final_frame = pd.DataFrame({'StockCode': codes, 
                                'Description': descriptions}) 
    # Switch order of columns around
    return final_frame[['StockCode', 'Description']] 



get_items_purchased(12346, product_train, customers_arr, 
                    products_arr, item_lookup)


rec_items(12346, product_train, user_vecs, 
          item_vecs, customers_arr, 
          products_arr, item_lookup,
          num_items = 10)




get_items_purchased(12353, product_train, 
                    customers_arr, products_arr, 
                    item_lookup)


rec_items(12353, product_train, user_vecs, 
          item_vecs, customers_arr, 
          products_arr, item_lookup,
          num_items = 10)




get_items_purchased(12361, product_train, 
                    customers_arr, products_arr, 
                    item_lookup)


rec_items(12361, product_train, user_vecs, 
          item_vecs, customers_arr, 
          products_arr, item_lookup,
          num_items = 10)








































