//
// Created by wyz on 20-12-25.
//

#ifndef B_SPLINE_BSPLINE_H
#define B_SPLINE_BSPLINE_H

#include <vector>
#include "Config.h"
class BSpline{
public:
    BSpline()
    :step(0.01f),order(4)
    {}

    virtual const std::vector<B_SPLINE_DATATYPE>& getInterpolationP(const std::vector<B_SPLINE_DATATYPE>& controlP)=0;
    virtual const std::vector<B_SPLINE_DATATYPE>& getApproximationP(std::vector<B_SPLINE_DATATYPE>& controlP)=0;
    void setupStep(float step){this->step=step;}
    void setupOrder(int order){this->order=order;};
protected:
    float step;
    int order;
    std::vector<B_SPLINE_DATATYPE> knots;
};


#endif //B_SPLINE_BSPLINE_H
