#pragma once
#include "circularbuffer.hpp"
#include <functional>
#include <limits>
#include <memory>
#include <mkl_types.h>
namespace mkl_wrapper {
class mkl_sparse_mat;
}

namespace reordering {

enum STATE { serial, parallel };

class BFS {
public:
  using FN = typename std::function<bool(
      mkl_wrapper::mkl_sparse_mat const *const, int, int, MKL_INT &, MKL_INT &,
      std::vector<MKL_INT> &, std::vector<MKL_INT> &)>;

  BFS(FN fn) : _fn{fn} {}
  template <bool LASTLEVEL = false>
  bool operator()(mkl_wrapper::mkl_sparse_mat const *const mat,
                  const MKL_INT s) {
    return _fn(mat, s, _shortCut, _height, _width, _levels, _lastLevel);
  }

  const std::vector<MKL_INT> &getLevels() const { return _levels; }

  MKL_INT getHeight() const { return _height; }
  MKL_INT getWidth() const { return _width; }

  void setShortCut(const MKL_INT sc) { _shortCut = sc; }

  const std::vector<MKL_INT> &getLastLevel() const { return _lastLevel; }

private:
  FN _fn;
  std::vector<MKL_INT> _lastLevel;
  std::vector<MKL_INT> _levels;
  MKL_INT _height;
  MKL_INT _width;
  MKL_INT _shortCut{std::numeric_limits<int>::max()};
};
template <bool LASTLEVEL = false>
bool BFS_Fn(mkl_wrapper::mkl_sparse_mat const *const mat, int source,
            int shortCut, MKL_INT &level, MKL_INT &width,
            std::vector<MKL_INT> &levels, std::vector<MKL_INT> &lastLevel);

template <bool LASTLEVEL = false, bool RECORDLEVEL = true>
bool PBFS_Fn(mkl_wrapper::mkl_sparse_mat const *const mat, int source,
             int shortCut, MKL_INT &level, MKL_INT &width,
             std::vector<MKL_INT> &levels, std::vector<MKL_INT> &lastLevel);
} // namespace reordering