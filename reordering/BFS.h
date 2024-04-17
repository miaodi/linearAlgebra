#pragma once
#include "circularbuffer.hpp"
#include <functional>
#include <memory>
#include <mkl_types.h>

namespace mkl_wrapper {
class mkl_sparse_mat;
}

namespace reordering {

enum STATE { serial, parallel };

class BFS {
public:
  using FN = typename std::function<void(
      mkl_wrapper::mkl_sparse_mat const *const, int, MKL_INT &,
      std::vector<MKL_INT> &, std::vector<MKL_INT> &)>;
  BFS(FN fn) : _fn{fn} {}
  template <bool LASTLEVEL = false>
  void operator()(mkl_wrapper::mkl_sparse_mat const *const mat,
                  const MKL_INT s) {
    _fn(mat, s, _level, _levels, _lastLevel);
  }

  const std::vector<MKL_INT> &getLevels() const { return _levels; }

  MKL_INT getLevel() const { return _level; }

  const std::vector<MKL_INT> &getLastLevel() const { return _lastLevel; }

private:
  FN _fn;
  std::vector<MKL_INT> _lastLevel;
  std::vector<MKL_INT> _levels;
  MKL_INT _level;
};
template <bool LASTLEVEL = false>
void BFS_Fn(mkl_wrapper::mkl_sparse_mat const *const mat, int source,
            MKL_INT &level, std::vector<MKL_INT> &levels,
            std::vector<MKL_INT> &lastLevel);

template <bool LASTLEVEL = false, bool RECORDLEVEL = true>
void PBFS_Fn(mkl_wrapper::mkl_sparse_mat const *const mat, int source,
             MKL_INT &level, std::vector<MKL_INT> &levels,
             std::vector<MKL_INT> &lastLevel);
} // namespace reordering