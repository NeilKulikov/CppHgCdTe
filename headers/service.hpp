#ifndef SERVICE
#define SERVICE

inline double lin(double a, double b, double x){
    return a * x + b * (1. - x);
};

template<typename T1, typename T2>
inline std::pair<T2, T1> pair_swap(const std::pair<T1, T2>& p){
    return {p.second, p.first};
};

#endif