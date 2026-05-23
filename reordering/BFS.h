#pragma once
#include "circularbuffer.hpp"
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace reordering {

enum STATE { serial, parallel };

template <typename ROWTYPE = int, typename COLTYPE = int,
          typename VALTYPE = double>
class BFS {
public:
  using FN = typename std::function<bool(
      COLTYPE, ROWTYPE const *, COLTYPE const *, VALTYPE const *, COLTYPE,
      COLTYPE, COLTYPE &, COLTYPE &, std::vector<COLTYPE> &,
      std::vector<COLTYPE> &)>;

  BFS(FN fn) : _fn{fn} {}
  template <bool LASTLEVEL = false>
  bool operator()(const COLTYPE rows, ROWTYPE const *const ai,
                  COLTYPE const *const aj, VALTYPE const *const av,
                  const COLTYPE source) {
    return _fn(rows, ai, aj, av, source, _shortCut, _height, _width, _levels,
               _lastLevel);
  }

  const std::vector<COLTYPE> &getLevels() const { return _levels; }

  std::vector<COLTYPE> &getLevels() { return _levels; }

  // number of levels of BFS
  COLTYPE getHeight() const { return _height; }

  // the max width of all levels
  COLTYPE getWidth() const { return _width; }

  void setShortCut(const COLTYPE sc) { _shortCut = sc; }

  const std::vector<COLTYPE> &getLastLevel() const { return _lastLevel; }

private:
  FN _fn;
  std::vector<COLTYPE> _lastLevel;
  std::vector<COLTYPE> _levels;
  COLTYPE _height;
  COLTYPE _width;
  COLTYPE _shortCut{std::numeric_limits<COLTYPE>::max()};
};

template <bool LASTLEVEL = false, typename ROWTYPE = int,
          typename COLTYPE = int, typename VALTYPE = double>
bool BFS_Fn(COLTYPE rows, ROWTYPE const *ai, COLTYPE const *aj,
            VALTYPE const *av, COLTYPE source, COLTYPE shortCut,
            COLTYPE &level, COLTYPE &width, std::vector<COLTYPE> &levels,
            std::vector<COLTYPE> &lastLevel);

template <bool LASTLEVEL = false, bool RECORDLEVEL = true,
          typename ROWTYPE = int, typename COLTYPE = int,
          typename VALTYPE = double>
bool PBFS_Fn(COLTYPE rows, ROWTYPE const *ai, COLTYPE const *aj,
             VALTYPE const *av, COLTYPE source, COLTYPE shortCut,
             COLTYPE &level, COLTYPE &width, std::vector<COLTYPE> &levels,
             std::vector<COLTYPE> &lastLevel);
} // namespace reordering
