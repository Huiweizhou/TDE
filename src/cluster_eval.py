from sklearn import metrics

def cluster_metric(X, labels_true, labels):

	homogeneity_score = metrics.homogeneity_score(labels_true, labels)
	completeness_score = metrics.completeness_score(labels_true, labels)
	v_measure_score = metrics.v_measure_score(labels_true, labels)
	adjusted_rand_score = metrics.adjusted_rand_score(labels_true, labels)
	adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels)
	silhouette_score = metrics.silhouette_score(X, labels)