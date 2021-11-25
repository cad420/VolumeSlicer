//
// Created by wyz on 20-12-25.
//

#include "BSplineCurve.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <Eigen/Core>
#include <Eigen/LU>

const std::vector<B_SPLINE_DATATYPE> &BSplineCurve::getInterpolationP(const std::vector<B_SPLINE_DATATYPE> &controlP) {
    const size_t n = controlP.size() / 3;
    knots.resize(n + order);
    for (size_t i = 0; i < order; i++)
        knots[i] = 0.f;
    for (size_t i = n; i < n + order; i++)
        knots[i] = 1.f;
    for (size_t i = order; i < n; i++)
        knots[i] = (i - order + 1) * 1.f / (n - order + 1);

    std::vector<float> b_spline_base;
    Eigen::MatrixXf N(n, n);
    Eigen::MatrixXf D(n, 3);
    for (size_t i = 0; i < n; i++) {
        D(i, 0) = controlP[i * 3];
        D(i, 1) = controlP[i * 3 + 1];
        D(i, 2) = controlP[i * 3 + 2];
    }
    std::vector<float> t_k;
    for (size_t i = 0; i < n; i++)
        t_k.push_back(i * 1.f / (n - 1));

    for (size_t t_i = 0; t_i < t_k.size(); t_i++) {
        b_spline_base.clear();
        b_spline_base.assign(n + order, 0.0);
        for (size_t i = 0; i < b_spline_base.size() - 1; i++) {
            if (t_k[t_i] == 1.f) {
                if (t_k[t_i] >= knots[i] && t_k[t_i] <= knots[i + 1]) {
                    b_spline_base[i] = 1.f;
                    break;
                }
            } else if (t_k[t_i] >= knots[i] && t_k[t_i] < knots[i + 1]) {
                b_spline_base[i] = 1.f;
            }
        }
        for (int k = 2; k <= order; k++) {
            for (size_t i = 0; i < n + order; i++) {
                float n_ik, n_i1k;
                if (i + k - 1 >= n + order || (knots[i + k - 1] - knots[i]) == 0.f)
                    n_ik = 0.f;
                else
                    n_ik = (t_k[t_i] - knots[i]) / (knots[i + k - 1] - knots[i]) *
                           b_spline_base[i];
                if (i + k >= n + order || (knots[i + k] - knots[i + 1]) == 0.f)
                    n_i1k = 0.f;
                else
                    n_i1k = (knots[i + k] - t_k[t_i]) / (knots[i + k] - knots[i + 1]) *
                            b_spline_base[i + 1];
                b_spline_base[i] = n_ik + n_i1k;
            }

        }
        for (size_t i = 0; i < n; i++) {
            N(t_i, i) = b_spline_base[i];
        }
    }

    Eigen::MatrixX3f P = N.lu().solve(D);
    std::vector<B_SPLINE_DATATYPE> control_points;
    control_points.reserve(controlP.size());
    for (size_t i = 0; i < P.rows(); i++) {
        control_points.push_back(P(i, 0));
        control_points.push_back(P(i, 1));
        control_points.push_back(P(i, 2));
    }
    return BaseFuncMethod(control_points);
}

const std::vector<B_SPLINE_DATATYPE> &BSplineCurve::getApproximationP(std::vector<B_SPLINE_DATATYPE> &controlP) {
    const size_t n = controlP.size() / 3;
    knots.resize(n + order);
    for (size_t i = 0; i < order; i++)
        knots[i] = 0.f;
    for (size_t i = n; i < n + order; i++)
        knots[i] = 1.f;
    for (size_t i = order; i < n; i++)
        knots[i] = (i - order + 1) * 1.f / (n - order + 1);

    std::vector<float> b_spline_base;
    Eigen::MatrixXf N(n - 2, h - 2);
    Eigen::MatrixXf D(n, 3);//control points
    Eigen::MatrixXf Qk(n, 3);
    Eigen::MatrixXf Q(h - 2, 3);
    for (size_t i = 0; i < n; i++) {
        D(i, 0) = controlP[i * 3 + 0];
        D(i, 1) = controlP[i * 3 + 1];
        D(i, 2) = controlP[i * 3 + 2];
    }

    std::vector<float> t_k;
    t_k.reserve(n);
    for (size_t i = 0; i < n; i++)
        t_k.push_back(i * 1.f / (n - 1));

    for (size_t t_i = 1; t_i < t_k.size() - 1; t_i++) {
        b_spline_base.clear();
        b_spline_base.assign(h + order, 0.0);
        for (size_t i = 0; i < b_spline_base.size() - 1; i++) {
            if (t_k[t_i] == 1.f) {
                if (t_k[t_i] >= knots[i] && t_k[t_i] <= knots[i + 1]) {
                    b_spline_base[i] = 1.f;
                    break;
                }
            } else if (t_k[t_i] >= knots[i] && t_k[t_i] < knots[i + 1]) {
                b_spline_base[i] = 1.f;
            }
        }
        for (int k = 2; k <= order; k++) {
            for (size_t i = 0; i < h + order; i++) {
                float n_ik, n_i1k;
                if (i + k - 1 >= h + order || (knots[i + k - 1] - knots[i]) == 0.f)
                    n_ik = 0.f;
                else
                    n_ik = (t_k[t_i] - knots[i]) / (knots[i + k - 1] - knots[i]) * b_spline_base[i];
                if (i + k >= h + order || (knots[i + k] - knots[i + 1]) == 0.f)
                    n_i1k = 0.f;
                else
                    n_i1k = (knots[i + k] - t_k[t_i]) / (knots[i + k] - knots[i + 1]) * b_spline_base[i + 1];
                b_spline_base[i] = n_ik + n_i1k;
            }

        }
        for (size_t i = 1; i < h - 1; i++)
            N(t_i - 1, i - 1) = b_spline_base[i];
        Qk(t_i, 0) = D(t_i, 0) - b_spline_base[0] * D(0, 0) - b_spline_base[h - 1] * D(n - 1, 0);
        Qk(t_i, 1) = D(t_i, 1) - b_spline_base[0] * D(0, 1) - b_spline_base[h - 1] * D(n - 1, 1);
        Qk(t_i, 2) = D(t_i, 2) - b_spline_base[0] * D(0, 2) - b_spline_base[h - 1] * D(n - 1, 2);
    }
    for (size_t i = 0; i < h - 2; i++) {
        Q(i, 0) = Q(i, 1) = Q(i, 2) = 0.f;
        for (size_t k = 0; k < n - 2; k++) {
            Q(i, 0) += N(k, i) * Qk(k + 1, 0);
            Q(i, 1) += N(k, i) * Qk(k + 1, 1);
            Q(i, 2) += N(k, i) * Qk(k + 1, 2);
        }
    }
    auto Nt_N = N.transpose() * N;
    Eigen::MatrixX3f P = Nt_N.lu().solve(Q);
    std::vector<B_SPLINE_DATATYPE> control_points;
    control_points.reserve(h * 3);
    control_points.push_back(D(0, 0));
    control_points.push_back(D(0, 1));
    control_points.push_back(D(0, 2));
    for (size_t i = 0; i < P.rows(); i++) {
        control_points.push_back(P(i, 0));
        control_points.push_back(P(i, 1));
        control_points.push_back(P(i, 2));
    }
    control_points.push_back(D(n - 1, 0));
    control_points.push_back(D(n - 1, 1));
    control_points.push_back(D(n - 1, 2));
    return BaseFuncMethod(control_points);
}

const std::vector<B_SPLINE_DATATYPE> &BSplineCurve::BaseFuncMethod(std::vector<float> &controlP) {
    this->interpolationP.clear();
    size_t n = controlP.size() / 3;
    knots.resize(n + order);
    for (size_t i = 0; i < order; i++)
        knots[i] = 0.f;
    for (size_t i = n; i < n + order; i++)
        knots[i] = 1.f;
    for (size_t i = order; i < n; i++)
        knots[i] = (i - order + 1) * 1.f / (n - order + 1);

    std::vector<float> b_spline_base;
    // t sample at [0,1)
    this->interpolationP.reserve(1.0 / step * 3 + 3);
    for (float t = 0; t < 1.0f; t += step) {
        float x = 0.f, y = 0.f, z = 0.f;
        b_spline_base.clear();
        b_spline_base.assign(n + order, 0.f);
        for (size_t i = 0; i < b_spline_base.size() - 1; i++) {
            if (t >= knots[i] && t < knots[i + 1]) {
                b_spline_base[i] = 1.f;
            }
        }
        for (int k = 2; k <= order; k++) {
            for (size_t i = 0; i < n + order; i++) {
                float n_ik, n_i1k;
                if (i + k - 1 >= n + order || (knots[i + k - 1] - knots[i]) == 0.f)
                    n_ik = 0.f;
                else
                    n_ik = (t - knots[i]) / (knots[i + k - 1] - knots[i]) * b_spline_base[i];
                if (i + k >= n + order || (knots[i + k] - knots[i + 1]) == 0.f)
                    n_i1k = 0.f;
                else
                    n_i1k = (knots[i + k] - t) / (knots[i + k] - knots[i + 1]) * b_spline_base[i + 1];
                b_spline_base[i] = n_ik + n_i1k;
            }

        }
        for (size_t i = 0; i < n; i++) {
            x += controlP[i * 3 + 0] * b_spline_base[i];
            y += controlP[i * 3 + 1] * b_spline_base[i];
            z += controlP[i * 3 + 2] * b_spline_base[i];
        }
        this->interpolationP.push_back(x);
        this->interpolationP.push_back(y);
        this->interpolationP.push_back(z);
    }

    return this->interpolationP;
}

const std::vector<B_SPLINE_DATATYPE> &BSplineCurve::DeBoor_Cox(std::vector<float> &controlP) {
    interpolationP.clear();
    size_t n = controlP.size() / 3;
    knots.resize(n + order);
    for (size_t i = 0; i < order; i++)
        knots[i] = 0.f;
    for (size_t i = n; i < n + order; i++)
        knots[i] = 1.f;
    for (size_t i = order; i < n; i++)
        knots[i] = (i - order + 1) * 1.f / (n - order + 1);
//    for(size_t i=0;i<knots.size();i++)
//        knots[i]=i*1.f/(knots.size()-1);
//    for(size_t i=0;i<knots.size();i++)
//        std::cout<<knots[i]<<" ";
//    std::cout<<std::endl;

    this->interpolationP.reserve(1.0 / step * 3 + 3);

    std::vector<B_SPLINE_DATATYPE> control_p;
    control_p.resize(order * 3);

    for (float t = 0.f; t < 1.0f; t += step) {
        int l;
        for (size_t i = 0; i < knots.size() - 1; i++) {
            if (t >= knots[i] && t < knots[i + 1]) {
                l = i;
                break;
            }
        }
        assert(l < knots.size());
//        std::cout<<"l: "<<l<<std::endl;
        for (size_t i = 0; i < order * 3; i++) {
            control_p[i] = controlP[i + 3 * (l - order + 1)];
        }

        // calculate order-1=degree times
        for (int k = 1; k < order; k++) {
//            std::cout<<"k: "<<k<<std::endl;
            for (int i = l; i >= l - order + 1 + k; i--) {
                if (i - 1 < 0) continue;
                int j = i - (l - order + 1);

                float tij = (t - knots[i]) / (knots[i + order - k] - knots[i]);
//                std::cout<<"t_i_j: "<<tij<<std::endl;
                control_p[j * 3 + 0] = (1.f - tij) * control_p[(j - 1) * 3 + 0]
                                       + tij * control_p[j * 3 + 0];
                control_p[j * 3 + 1] = (1.f - tij) * control_p[(j - 1) * 3 + 1]
                                       + tij * control_p[j * 3 + 1];
                control_p[j * 3 + 2] = (1.f - tij) * control_p[(j - 1) * 3 + 2]
                                       + tij * control_p[j * 3 + 2];
//                std::cout<<control_p[j*3+0]<<" "
//                        <<control_p[j*3+1]<<" "
//                        <<control_p[j*3+2]<<std::endl;
            }
        }
//        std::cout<<t<<" ";
//        std::cout<<control_p[(order-1)*3+0]<<" "
//                <<control_p[(order-1)*3+1]<<" "
//                <<control_p[(order-1)*3+2]<<std::endl;
        this->interpolationP.push_back(control_p[(order - 1) * 3 + 0]);
        this->interpolationP.push_back(control_p[(order - 1) * 3 + 1]);
        this->interpolationP.push_back(control_p[(order - 1) * 3 + 2]);
    }

    control_p.clear();
//    std::cout<<"size "<<interpolationP.size()/3<<std::endl;
    return this->interpolationP;
}

void BSplineCurve::setupApproximationH(size_t h) {
    this->h = h;
}


