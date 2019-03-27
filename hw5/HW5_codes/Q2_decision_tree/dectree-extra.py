    @ray.remote
    def eval_split(self, x, y, feature, value):
        _, y_left, _, y_right = DecisionTree.split(x, y, feature, value)
        if len(y_left) > 0 and len(y_right) > 0:
            ig = DecisionTree.info_gain(y, y_left, y_right)
            return (ig, feature, value)

    def best_split(self, x, y):
        ray.init()  
        num_features = x.shape[1]
        features = range(num_features)
        threshs = [value for feature in features for value in np.unique(x[:,feature])]
        results = []
        for f,v in itertools.product(features,threshs):
            results.append(self.eval_split.remote(self,x,y,f,v))
            
        #results = ray.get([self.eval_split.remote(self,x,y,f,v) for f,v in itertools.product(features,threshs)])
        best_result = max(results,key=itemgetter(0))
        best_feature, best_thresh = best_result[1], best_result[2]
        return best_feature, best_thresh