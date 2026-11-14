#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    int32_t n_trees;
    int32_t *tree_offsets;
    int32_t *children_left;
    int32_t *children_right;
    int32_t *feature;
    double *threshold;
    double *leaf_depth;
    double c_max;
    double offset;
} IForestModel;

IForestModel *iforest_create(
    int32_t n_trees,
    const int32_t *tree_offsets,
    const int32_t *children_left,
    const int32_t *children_right,
    const int32_t *feature,
    const double *threshold,
    const double *leaf_depth,
    int32_t total_nodes,
    double c_max,
    double offset
) {
    IForestModel *m = (IForestModel *)malloc(sizeof(IForestModel));
    if (!m) return NULL;
    m->n_trees = n_trees;
    m->c_max = c_max;
    m->offset = offset;

    m->tree_offsets = (int32_t *)malloc((size_t)n_trees * sizeof(int32_t));
    m->children_left = (int32_t *)malloc((size_t)total_nodes * sizeof(int32_t));
    m->children_right = (int32_t *)malloc((size_t)total_nodes * sizeof(int32_t));
    m->feature = (int32_t *)malloc((size_t)total_nodes * sizeof(int32_t));
    m->threshold = (double *)malloc((size_t)total_nodes * sizeof(double));
    m->leaf_depth = (double *)malloc((size_t)total_nodes * sizeof(double));

    if (!m->tree_offsets || !m->children_left || !m->children_right ||
        !m->feature || !m->threshold || !m->leaf_depth) {
        free(m->tree_offsets);
        free(m->children_left);
        free(m->children_right);
        free(m->feature);
        free(m->threshold);
        free(m->leaf_depth);
        free(m);
        return NULL;
    }

    for (int32_t i = 0; i < n_trees; i++) m->tree_offsets[i] = tree_offsets[i];
    for (int32_t i = 0; i < total_nodes; i++) {
        m->children_left[i] = children_left[i];
        m->children_right[i] = children_right[i];
        m->feature[i] = feature[i];
        m->threshold[i] = threshold[i];
        m->leaf_depth[i] = leaf_depth[i];
    }
    return m;
}

void iforest_free(IForestModel *m) {
    if (!m) return;
    free(m->tree_offsets);
    free(m->children_left);
    free(m->children_right);
    free(m->feature);
    free(m->threshold);
    free(m->leaf_depth);
    free(m);
}

double iforest_decision_one(const IForestModel *m, const float *x) {
    double depth_sum = 0.0;
    const int32_t n_trees = m->n_trees;
    const int32_t *tree_offsets = m->tree_offsets;
    const int32_t *c_left = m->children_left;
    const int32_t *c_right = m->children_right;
    const int32_t *feat = m->feature;
    const double *thresh = m->threshold;
    const double *ldepth = m->leaf_depth;

    for (int32_t t = 0; t < n_trees; t++) {
        int32_t node = tree_offsets[t];
        int32_t left = c_left[node];
        while (left != -1) {
            if ((double)x[feat[node]] <= thresh[node]) {
                node = left;
            } else {
                node = c_right[node];
            }
            left = c_left[node];
        }
        depth_sum += ldepth[node];
    }

    double denom = (double)n_trees * m->c_max;
    double score = (denom > 0.0) ? -exp2(-depth_sum / denom) : -0.5;
    return score - m->offset;
}
