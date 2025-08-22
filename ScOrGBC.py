import numpy as np
import time
import copy
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.cluster._kmeans')

def create_initial_spheres(X, y, centers):
    if centers.shape[0] == 0: return []
    from sklearn.metrics.pairwise import pairwise_distances_argmin_min
    cluster_assignments, distances = pairwise_distances_argmin_min(X, centers)
    spheres = []
    for i in range(len(centers)):
        points_in_cluster_indices = np.where(cluster_assignments == i)[0]
        if len(points_in_cluster_indices) == 0: continue

        majority_label = Counter(y[points_in_cluster_indices]).most_common(1)[0][0]
        # print('====>',y[points_in_cluster_indices])
        radius = np.max(distances[cluster_assignments == i]) if len(points_in_cluster_indices) > 0 else 0

        spheres.append({
            'id': i,
            'center': centers[i],
            'radius': radius,
            'class': majority_label,
            'point_indices': points_in_cluster_indices
        })
    return spheres


def iterative_merge_spheres(spheres, max_iters=10, tolerance=1e-6):
    if not spheres or len(spheres) < 2: return copy.deepcopy(spheres)
    adjusted_spheres = copy.deepcopy(spheres)
    num_spheres = len(adjusted_spheres)
    for iteration in range(max_iters):
        adjustments = np.zeros(num_spheres)
        for i in range(num_spheres):
            sphere_i = adjusted_spheres[i]
            max_overlap = 0.0
            neighbor_j = -1
            for j in range(num_spheres):
                if i == j: continue
                sphere_j = adjusted_spheres[j]
                if sphere_i['class'] == sphere_j['class']: continue
                dist = np.linalg.norm(sphere_i['center'] - sphere_j['center'])
                overlap = (sphere_i['radius'] + sphere_j['radius']) - dist
                if overlap > max_overlap:
                    max_overlap, neighbor_j = overlap, j
            if max_overlap > 0:
                neighbor_s = adjusted_spheres[neighbor_j]
                total_r = sphere_i['radius'] + neighbor_s['radius']
                shrink_amount = max_overlap * (sphere_i['radius'] / total_r)
                adjustments[i] = shrink_amount
        if np.sum(adjustments) < tolerance: break

        for i in range(num_spheres):
            adjusted_spheres[i]['radius'] = adjusted_spheres[i]['radius'] - adjustments[i]
    return adjusted_spheres


def optimize_final_radii_local(spheres, X_train, y_train, g_factor, n_steps=20):
    if not spheres: return []
    final_spheres = copy.deepcopy(spheres)

    for sphere in final_spheres:
        r_max = sphere['radius']
        if r_max <= 1e-6: continue

        r_min = r_max / 2.0
        initial_member_indices = sphere.get('point_indices')
        if initial_member_indices is None or len(initial_member_indices) == 0:
            continue

        X_cluster_members = X_train[initial_member_indices]
        dists = np.linalg.norm(X_cluster_members - sphere['center'], axis=1)

        best_r, max_score = r_max, -1.0
        for r_cand in np.linspace(r_min, r_max, n_steps):
            num_covered = np.sum(dists <= r_cand)
            score = num_covered * np.exp(-g_factor * r_cand)
            if score > max_score:
                max_score, best_r = score, r_cand
        sphere['radius'] = best_r
    return final_spheres

def predict(X_test, spheres, classes):
    if not spheres: return np.full(len(X_test), classes[0]) if classes.any() else np.array([])
    predictions = []
    for x in X_test:
        metrics = [np.linalg.norm(x - s['center']) - s['radius']for s in spheres]
        best_idx = np.argmin(metrics)
        predictions.append(spheres[best_idx]['class'])
    return np.array(predictions)

_experiment_results_cache = {}


def _run_full_experiment_loop(X_train, y_train, X_test, y_test):
    results_v1, results_v2, results_v3 = [], [], []
    scale_factors = np.arange(0.5, 10.5, 0.5)
    g_factors = range(1, 11)  # g 从 1 到 10
    classes = np.unique(np.concatenate((y_train, y_test)))

    for scale_factor in scale_factors:
        sf_rounded = round(scale_factor, 1)
        num_spheres_target = max(1, int(sf_rounded * np.sqrt(len(X_train))))
        if num_spheres_target>len(y_train):
            num_spheres_target=len(y_train)
        start_time_total = time.time()
        kmeans = KMeans(n_clusters=num_spheres_target, init='k-means++', random_state=42)
        kmeans.fit(X_train)
        spheres_v1 = create_initial_spheres(X_train, y_train, kmeans.cluster_centers_)
        time_v1 = time.time() - start_time_total
        y_pred_v1 = predict(X_test, spheres_v1, classes)
        results_v1.append({
            'scale_factor': sf_rounded, 'accuracy': accuracy_score(y_test, y_pred_v1),
            'f1_score': f1_score(y_test, y_pred_v1, average='weighted', zero_division=0),
            'num_spheres': len(spheres_v1), 'runtime': time_v1
        })
        spheres_v2 = iterative_merge_spheres(spheres_v1, max_iters=10)
        time_v2 = time.time() - start_time_total
        y_pred_v2 = predict(X_test, spheres_v2, classes)
        results_v2.append({
            'scale_factor': sf_rounded, 'accuracy': accuracy_score(y_test, y_pred_v2),
            'f1_score': f1_score(y_test, y_pred_v2, average='weighted', zero_division=0),
            'num_spheres': len(spheres_v2), 'runtime': time_v2
        })
        for g in g_factors:
            v3_opt_start_time = time.time()
            spheres_v3_candidate = optimize_final_radii_local(spheres_v2, X_train, y_train, g_factor=g)
            time_v3_candidate = time_v2 + (time.time() - v3_opt_start_time)

            y_pred_v3 = predict(X_test, spheres_v3_candidate, classes)
            acc = accuracy_score(y_test, y_pred_v3)
            f1 = f1_score(y_test, y_pred_v3, average='weighted', zero_division=0)
            results_v3.append({
                'scale_factor': sf_rounded,
                'g_factor': g, 
                'accuracy': acc,
                'f1_score': f1,
                'num_spheres': len(spheres_v3_candidate),
                'runtime': time_v3_candidate
            })

    return results_v1, results_v2, results_v3


def _format_results_to_tuples_v1_v2(results_list_of_dicts):
    if not results_list_of_dicts: return []
    return [(item['scale_factor'], item['accuracy'], item['f1_score'], item['num_spheres'], item['runtime']) for item in
            results_list_of_dicts]


def _format_results_to_tuples_v3(results_list_of_dicts):
    if not results_list_of_dicts: return []
    return [(item['scale_factor'], item['g_factor'], item['accuracy'], item['f1_score'], item['num_spheres'],
             item['runtime']) for item in
            results_list_of_dicts]


def get_or_run_all_experiments(X_train, y_train, X_test, y_test):
    cache_key = (
    X_train.tobytes(), y_train.tobytes(), X_test.tobytes(), y_test.tobytes(), "v_final_with_full_g_results")
    if cache_key not in _experiment_results_cache:
        _experiment_results_cache[cache_key] = _run_full_experiment_loop(X_train, y_train, X_test, y_test)
    return _experiment_results_cache[cache_key]

def ScOrGBC1(X_train, X_test, y_train, y_test):
    results_v1, _, _ = get_or_run_all_experiments(X_train, y_train, X_test, y_test)
    return _format_results_to_tuples_v1_v2(results_v1)


def ScOrGBC2(X_train, X_test, y_train, y_test):
    _, results_v2, _ = get_or_run_all_experiments(X_train, y_train, X_test, y_test)
    return _format_results_to_tuples_v1_v2(results_v2)


def ScOrGBC(X_train, X_test, y_train, y_test):
    _, _, results_v3 = get_or_run_all_experiments(X_train, y_train, X_test, y_test)
    return _format_results_to_tuples_v3(results_v3)
