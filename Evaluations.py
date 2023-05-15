import random
import numpy as np

class evaluation_metrics():
    
    def _precision(predictions , actuals, k = None):
        """
        Calculate the precision at k
        
        Returns: a list of precisions
        """
        precisions =[]

        for i in range(len(predictions)):
            
            prediction = predictions[i]

            if  k != None:
                prediction =  predictions[i][:k]
            
            score = 0
            for j in prediction:
                if j in actuals[i]:
                    score+=1
            if len(prediction) != 0:
                precisions.append(score/len(prediction))
            else:
                precisions.append(0)
        return precisions
        

    def _recall(predictions , actuals, k = None):
        
        """
        Calculate the precision at k
        
        Returns: a list of recalls
        """
        recalls =[]
        
        for i in range(len(predictions)):
            
            prediction =  predictions[i]
            
            if  k != None:
                prediction =  predictions[i][:k]
            
            score = 0
            for j in range(len(prediction)):
                if prediction[j] in actuals[i]:
                    score+=1
            recalls.append(score/len(actuals[i]))
        
        return recalls
    
    def _mrr(predictions, actuals):
        """
        Calculate the mean reciprocal rank (MRR) for a set of predictions and actual values.
        
        Parameters:
        predictions (list of lists): A list of predicted rankings sorted by probability.
        actual (list of lists): A list of actual rankings sorted by relevance.
        
        Returns:    
        float: A list of MRR scores.
        """
        mrr_list = []
        for i in range(len(predictions)):
            reciprocal_rank = 0
            if actuals[i][0] in predictions[i]:
                reciprocal_rank = 1/ (predictions[i].index(actuals[i][0]) + 1)
            mrr_list.append(reciprocal_rank)
        return mrr_list

    def _map( predictions , actuals, k=None):
        """
        Calculate the mean average precision (MAP) for a set of queries.

        Parameters:
        actual (list of sets or lists): A list of sets or lists of the actual relevant items for each query.
        predicted (list of lists): A list of lists of predicted items for each query.
        k (int): The maximum number of predicted items to consider for each query.

        Returns:
        float: A list of MAP scores.
        """
        
        map_list = []
        

        for i in range(len(predictions)):
            
            ap_list = []
            hit = 0 
            cnt = 0 
            
            prediction =  predictions[i]
            
            if k != None:
                prediction =  predictions[i][:k]
            
            
            for j in prediction:
                if j in actuals[i]:
                    hit+=1
                    cnt+=1
                    ap_list.append(hit/cnt)
                else:
                    cnt+=1
            if len(ap_list) != 0:
                map_list.append(np.mean(ap_list))
            else:
                map_list.append(0)
        
        return map_list
    
    def _dcg_ndcg( predictions , actuals, rel ,k=None):
        """
        Calculate the DCG@k , NDCG@k for a set of queries.

        Parameters:
        actual (list of sets or lists): A list of sets or lists of the actual relevant items for each query.
        predicted (list of lists): A list of lists of predicted items for each query.
        k (int): The maximum number of predicted items to consider for each query.

        Returns:
        float: A list of DCG , NDCG scores.
        """
        dcg_list = []
        ndcg_list = []
        
        for i in range(len(predictions)):
            dcg =0
            idcg =0
            
            prediction = predictions[i]
            
            if k != None:
                prediction = predictions[i][:k]
            
            for j in range(len(actuals[i])):
                if actuals[i][j] in prediction:
                    rank = prediction.index(actuals[i][j]) + 1
                    dcg += np.divide(float(rel[i][j]),np.log2(rank+1))
                idcg += np.divide(float(rel[i][j]),np.log2((j+1)+1))
            dcg_list.append(dcg)
            if np.divide(dcg,idcg) > 1:
                print(rel[i], prediction,actuals[i]  )
                print(i,dcg,idcg  , 'Wrong !!!')
            ndcg_list.append(np.divide(dcg,idcg))
            
        return dcg_list , ndcg_list
                
