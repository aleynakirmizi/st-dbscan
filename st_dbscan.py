class st_dbSCAN():
    def __init__(self, df, eps1, eps2, minpt,epsilon):
        #df: cleaned dataset
        #eps1 : maximum difference between 2 headline
        # eps 2 : maximum temporal diff value(timestamp)
        #minpt : minimum number of points within eps1 and eps2
        self.df = df
        self.eps1 = eps1
        self.eps2 = eps2
        self.minpt = minpt
        self.epsilon = epsilon

    def fit_transform(self, df):
        cluster_label = 0
        noise = -1
        unmarked = 999999
        stack = []
        df["cluster"] = unmarked #build a new column called cluster and initialize each point in the dataset as unmarked
        for index, _ in df.iterrows(): # for each index in dataframe
            if df.loc[index]["cluster"] == unmarked: # find the unmarked points of relevant index
                neighborhood = self.retrieve_neighbors(df, index) # find neighboors of relevant index
                if len(neighborhood) < self.minpt: #if number of neighborhood found is less than minpt
                    df.at[index, "cluster"] = noise #access the index/cluster (row/column) label pair and assign it as noise
                else: # if number of neighborhood is more or equal  to minpt then it finds a core point
                    cluster_label += 1 #increase cluster_label
                    df.at[index, "cluster"] = cluster_label  # assign a label to core point
                    for neighbor in neighborhood:  # go inside the neighboorhood list.
                        df.at[neighbor, "cluster"] = cluster_label # assign core's label to its neighborhood
                        stack.append(neighbor) #push the index into the stack
                    while stack:
                        current_object = stack.pop()  # selects and removes the last index from the stack
                        new_neighborhood = self.retrieve_neighbors(df, current_object)# finds neighboors of relevant index

                        if len(new_neighborhood) >= self.minpt:
                            for new_neig_index in new_neighborhood:
                                new_neig_cluster_label = df.loc[new_neig_index]["cluster"]
                                if new_neig_cluster_label == unmarked and abs(self.cluster_avg(df,cluster_label)-df.iloc[new_neig_index,1])<self.epsilon:
                                    df.at[new_neig_index, "cluster"] = cluster_label
                                    stack.append(new_neig_index)
        return df

    def cluster_avg(self, df, cluster_label):
        cluster = df[(df.iloc[:, 2] == cluster_label)]
        df_mean = cluster.mean()
        return df_mean[0]

    def time_filter(self, place, df):
        # filter by time
        mintime = place[1] - self.eps2
        maxtime = place[1] + self.eps2
        return df[((df.iloc[:, 1] >= mintime) & (df.iloc[:, 1] <= maxtime))]

    def retrieve_neighbors(self, df, index_point):
        neigbor_index_list = []
        place = df.loc[index_point]
        df = self.time_filter(place, df)
        for index, _ in df.iterrows():
            if index != index_point:
                intersection = len(list(set(place[0]).intersection(df.loc[index][0])))
                union = (len(place[0]) + len(df.loc[index][0])) - intersection
                try:
                    similarity = float(intersection) / union
                except ZeroDivisionError:
                    similarity = 0
                dist = 1 - similarity
                if dist <= self.eps1:
                    neigbor_index_list.append(index)
        return neigbor_index_list