#include "MinimumDegree.hpp"
#include <algorithm>
#include <assert.h>
namespace reordering {

template <typename COLTYPE>
template <typename ROWTYPE>
void QuotientGraph<COLTYPE>::initialize(const COLTYPE nnodes, ROWTYPE const *ai,
                                        COLTYPE const *aj) {
  _nodes.resize(nnodes);
  _union_find.reset(nnodes); // reset union-find structure

  _degree_to_principle.clear();

  __temp1.reserve(nnodes);
  __temp2.reserve(nnodes); // temporary vectors for sorting

  const ROWTYPE base = ai[0];
  for (COLTYPE i = 0; i < nnodes; i++) {
    _nodes[i].adjacent_variables.reserve(ai[i + 1] - ai[i]);
    for (ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++) {
      COLTYPE j = aj[j_idx] - base;
      if (j == i)
        continue;
      _nodes[i].adjacent_variables.push_back(j);
    }
    _nodes[i].degree = _nodes[i].adjacent_variables.size();
    _degree_to_principle[_nodes[i].degree].insert(i);
    _nodes[i].simple_variables.push_back(i);
  }
}

template <typename COLTYPE>
template <typename ROWTYPE>
void QuotientGraph<COLTYPE>::operator()(const COLTYPE nnodes, ROWTYPE const *ai,
                                        COLTYPE const *aj, COLTYPE *iperm,
                                        COLTYPE *perm) {
  initialize(nnodes, ai, aj);
}

template <typename COLTYPE>
bool QuotientGraph<COLTYPE>::isDistinguishable(const COLTYPE i,
                                               const COLTYPE j) const {
  // diff in quotient graph
  if (_nodes[i].adjacent_variables.size() !=
      _nodes[j].adjacent_variables.size())
    return true;
  if (_nodes[i].adjacent_elements.size() != _nodes[j].adjacent_elements.size())
    return true;

  // check if they have the same adjacent variables
  size_t k = 0, l = 0;
  while (k < _nodes[i].adjacent_variables.size() &&
         l < _nodes[j].adjacent_variables.size()) {
    if (_nodes[i].adjacent_variables[k] != _nodes[j].adjacent_variables[l]) {
      if (_nodes[i].adjacent_variables[k] == j)
        k++;
      else if (_nodes[j].adjacent_variables[l] == i)
        l++;
      else
        return true; // different neighbours
    } else {
      k++;
      l++;
    }
  }
  if (k < _nodes[i].adjacent_variables.size()) {
    if (_nodes[i].adjacent_variables[k] == j)
      k++;
    else
      return true; // different neighbours
  }
  if (l < _nodes[j].adjacent_variables.size()) {
    if (_nodes[j].adjacent_variables[l] == i)
      l++;
    else
      return true; // different neighbours
  }
  if (k != _nodes[i].adjacent_variables.size() ||
      l != _nodes[j].adjacent_variables.size())
    return true; // different neighbours

  // check if they have the same adjacent elements
  k = 0;
  l = 0;
  while (k < _nodes[i].adjacent_elements.size() &&
         l < _nodes[j].adjacent_elements.size()) {
    if (_nodes[i].adjacent_elements[k] != _nodes[j].adjacent_elements[l]) {
      return true;
    } else {
      k++;
      l++;
    }
  }
  return false;
}

template <typename COLTYPE>
void QuotientGraph<COLTYPE>::updateNode(const COLTYPE i) {
  assert(i == _union_find.Find(i));
  principleVector(_nodes[i].simple_variables);
  principleVector(_nodes[i].adjacent_variables);
  principleVector(_nodes[i].adjacent_elements);
}

template <typename COLTYPE>
std::vector<COLTYPE> &QuotientGraph<COLTYPE>::getFillins(const COLTYPE i) {
  // TODO: check if really needed
  updateNode(i);

  __temp1.clear();
  __vectors.clear();
  __vectors.push_back(&_nodes[i].adjacent_variables);

  for(auto j: _nodes[i].adjacent_elements) {
    updateNode(j);
    __vectors.push_back(&_nodes[j].adjacent_variables);
  }

  mergeKVectors(__vectors, __temp1);
  vectorSubtract(__temp1, _nodes[i].simple_variables);
  return __temp1;
}

template <typename COLTYPE>
void QuotientGraph<COLTYPE>::merge(const COLTYPE i, const COLTYPE j) {
  auto principle_i = _union_find.Find(i);
  auto principle_j = _union_find.Find(j);
  if (principle_i == principle_j) {
    return; // already merged
  }
  auto parent = _union_find.Unite(principle_i, principle_j);
  if (principle_j != parent) {
    std::swap(principle_i, principle_j);
  }
  auto VectorMerge = [this](std::vector<COLTYPE> &v1,
                             const std::vector<COLTYPE> &v2) {
    if (v2.empty())
      return;
    if (v1.empty()) {
      v1 = v2;
      return;
    }
    auto it1 = v1.begin();
    auto it2 = v2.begin();
    while (it1 != v1.end() && it2 != v2.end()) {
      if (*it1 < *it2) {
        ++it1;
      } else if (*it1 > *it2) {
        it1 = v1.insert(it1, *it2++);
      } else {
        ++it2;
      }
    }
    while (it2 != v2.end()) {
      it1 = v1.insert(it1, *it2++);
    }
  };
}


template <typename COLTYPE>
void QuotientGraph<COLTYPE>::massElimination(const COLTYPE i) {
  auto& fill_ins = getFillins(i);
  for (auto j : fill_ins) {
    // remove redundant variables
    vectorSubtract(_nodes[j].adjacent_variables, fill_ins);
    vectorSubtract(_nodes[j].adjacent_variables, _nodes[i].simple_variables);

    // element absorption


  }
}



template <typename COLTYPE>
void QuotientGraph<COLTYPE>::principleVector(std::vector<COLTYPE> &vec) {
  __temp2.clear();
  for (auto i : vec) {
    __temp2.push_back(_union_find.Find(i));
  }
  std::sort(__temp2.begin(), __temp2.end());
  vec.clear();
  for (auto i : __temp2) {
    if (vec.empty() || vec.back() != i) {
      vec.push_back(i);
    }
  }
}

template class QuotientGraph<int>;
template void QuotientGraph<int>::operator()(const int nnodes, int const *ai,
                                             int const *aj, int *iperm,
                                             int *perm);

template void QuotientGraph<int>::initialize(const int nnodes, int const *ai,
                                             int const *aj);

} // namespace reordering