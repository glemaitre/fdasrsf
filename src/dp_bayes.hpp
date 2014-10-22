struct dp_result {
    uvec J;
    double NDist;
    vec q2LL;
};

dp_result dp_bayes(vec q1, vec q1L, vec q2L, int times, int cut);

