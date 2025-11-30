#pragma once

#include "Transformation.hpp"   
#include "matrix_utils.hpp"
#include <memory>
#include <vector>
#include <stdexcept>

namespace solver
{

/**
 * Linear solver system with matrix preconditioning transformations
 * 
 * TODO: Design and implement the linear solver system
 */
template <typename CSRMatrixType>
class LinearSolverSystem
{
public:
    using ROWTYPE = typename CSRMatrixType::ROWTYPE;
    using COLTYPE = typename CSRMatrixType::COLTYPE;
    using VALTYPE = typename CSRMatrixType::VALTYPE;
    using VecType = std::vector<VALTYPE>;
    using TransformPtr = std::shared_ptr<TransformationBase<CSRMatrixType, VecType>>;
    
    LinearSolverSystem() = default;

private:
};

} // namespace solver
