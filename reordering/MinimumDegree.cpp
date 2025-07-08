#include "MinimumDegree.hpp"
#include <algorithm>
#include <assert.h>
#include <limits> // Required for std::numeric_limits
namespace reordering {

template <typename COLTYPE, MinimumDegree MD>
template <typename ROWTYPE>
void QuotientGraph<COLTYPE, MD>::initialize(const COLTYPE nnodes,
                                            ROWTYPE const *ai,
                                            COLTYPE const *aj, COLTYPE *&perm) {
  using ObjectType = typename decltype(_pool)::value_type;
  const COLTYPE size = std::max((COLTYPE)100, (COLTYPE)(nnodes / 1000.));
  _pool.setObjectPrep([size](ObjectType *obj) {
    obj->reserve(size);
    obj->clear();
  });
  using CBType = typename decltype(_cb_pool)::value_type;
  _cb_pool.setObjectPrep([size](CBType *obj) {
    if (obj->size() < size)
      obj->resize(size);
    obj->clear();
  });

  _nodes.resize(nnodes);
  _union_find.reset(nnodes); // reset union-find structure
  _degree_to_principle.clear();
  const ROWTYPE base = ai[0];
  for (COLTYPE i = 0; i < nnodes; i++) {
    COLTYPE row_size = ai[i + 1] - ai[i];

    if (row_size < 2) {
      *perm++ = i + base;
      continue;
    }

    _nodes[i].adjacent_variables.reserve(row_size);
    for (ROWTYPE j_idx = ai[i] - base; j_idx < ai[i + 1] - base; j_idx++) {
      COLTYPE j = aj[j_idx] - base;
      if (j == i)
        continue;
      _nodes[i].adjacent_variables.push_back(j);
    }
    _nodes[i].degree = _nodes[i].adjacent_variables.size();
    _nodes[i].simple_variables.push_back(i);

    auto it = _degree_to_principle.try_emplace(_nodes[i].degree,
                                               std::move(_cb_pool.acquire()));
    it.first->second->nonOverwritePush(i);
    std::cout<<"Node " << i << " has degree " << _nodes[i].degree << std::endl;
    // _degree_to_principle[_nodes[i].degree].insert(i);
  }
  for(COLTYPE i = 0; i < nnodes; i++) {
    std::cout<<"Node " << i << " simple_variables: ";
    for(auto j : _nodes[i].simple_variables) {
      std::cout << j << " ";
    }
    std::cout << " degree: " << _nodes[i].degree << std::endl;
  }
}

template <typename COLTYPE, MinimumDegree MD>
template <typename ROWTYPE>
void QuotientGraph<COLTYPE, MD>::operator()(const COLTYPE nnodes,
                                            ROWTYPE const *ai,
                                            COLTYPE const *aj, COLTYPE *perm,
                                            COLTYPE *iperm) {
  const ROWTYPE base = ai[0];
  COLTYPE idx = 0;
  bool found;
  typename decltype(_degree_to_principle)::iterator it;
  initialize(nnodes, ai, aj, perm);
  while (!_degree_to_principle.empty()) {
    // std::cout << _degree_to_principle.size() << " " << _cb_pool.size() << " "
    //           << _pool.size() << std::endl;
    it = _degree_to_principle.begin();
    found = false;
    while (!it->second->isEmpty()) {
      idx = it->second->first();
      it->second->shift();
      if (_union_find.Find(idx) == idx && _nodes[idx].degree == it->first) {
        found = true;
        break; // found a valid principle
      }
    }
    if (!found) {
      _degree_to_principle.erase(it);
    } else {
      // std::cout << " Eliminating principle node " << idx << " with degree "
      //           << _nodes[idx].degree << std::endl;
      eliminatePrincipleNode(idx, perm);
    }
  }
}

template <typename COLTYPE, MinimumDegree MD>
bool QuotientGraph<COLTYPE, MD>::isDistinguishable(const COLTYPE i,
                                                   const COLTYPE j) const {
  // diff in quotient graph
  if (_nodes[i].adjacent_variables.size() !=
      _nodes[j].adjacent_variables.size())
    return true;
  if (_nodes[i].adjacent_elements.size() != _nodes[j].adjacent_elements.size())
    return true;
  // std::cout << "hello!\n";
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
  if (k < _nodes[i].adjacent_elements.size() ||
      l < _nodes[j].adjacent_elements.size())
    return true; // different neighbours
  return false;  // distinguishable
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::getFillins(const COLTYPE p,
                                            std::vector<COLTYPE> &Lp,
                                            std::vector<COLTYPE> &temp) {
  // TODO: check if really needed
  principleVector(_nodes[p].adjacent_variables, temp);

  __vectors.clear();
  __vectors.push_back(&_nodes[p].adjacent_variables);
  auto &adj_elements = _nodes[p].adjacent_elements;
  COLTYPE pos = 0;
  for (size_t i = 0; i < adj_elements.size(); ++i) {
    COLTYPE j = adj_elements[i];
    if (_nodes[j].degree != element) {
      continue;
    }
    principleVector(_nodes[j].adjacent_variables, temp);
    __vectors.push_back(&_nodes[j].adjacent_variables);
    adj_elements[pos++] = j; // keep only element nodes
  }
  adj_elements.resize(pos); // remove non-element nodes
  assert(Lp.empty());
  mergeKVectors(__vectors, Lp, std::optional<COLTYPE>(p));
  // vectorSubtract(Lp, _nodes[p].simple_variables);
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::merge(const COLTYPE i, const COLTYPE j) {
  std::cout << "merging " << i << " and " << j << std::endl;
  assert(_union_find.Find(i) !=
         _union_find.Find(j)); // cannot merge the same element
  std::vector<COLTYPE> temp;
  temp.reserve(_nodes[i].simple_variables.size() +
               _nodes[j].simple_variables.size());
  auto it = std::set_union(
      _nodes[i].simple_variables.begin(), _nodes[i].simple_variables.end(),
      _nodes[j].simple_variables.begin(), _nodes[j].simple_variables.end(),
      std::back_inserter(temp));
  assert(std::is_sorted(temp.begin(), temp.end()));
  assert(std::adjacent_find(temp.begin(), temp.end()) == temp.end());
  std::swap(_nodes[i].simple_variables, temp);
  _nodes[i].degree -= static_cast<COLTYPE>(_nodes[j].simple_variables.size());
  clearNode(j);
  assert(i == _union_find.Unite(i, j));
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::eliminatePrincipleNode(const COLTYPE p,
                                                        COLTYPE *&perm) {
  // std::cout << "eliminate " << p << std::endl;
  assert(p == _union_find.Find(p));
  auto Lp_ptr = massElimination(p);
  for (auto j : _nodes[p].simple_variables) {
    *perm++ = j;
  }

  std::cout << p << " fillins: ";
  for (auto i : *Lp_ptr) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  supervariableMerge(*Lp_ptr);
  toElementNode(p);
}

template <typename COLTYPE, MinimumDegree MD>
auto QuotientGraph<COLTYPE, MD>::massElimination(const COLTYPE p) {
  auto Lp = _pool.acquire();
  auto temp = _pool.acquire();
  getFillins(p, *Lp, *temp); // getFillins will replace simple_variables with
                             //  principle variables
  _nodes[p].adjacent_variables = *Lp;
  for (auto i : _nodes[p].adjacent_elements) {
    removeElementNode(i);
  }
  for (auto i : *Lp) {
    // remove redundant variables
    vectorSubtract(_nodes[i].adjacent_variables, *Lp, std::optional<COLTYPE>(p));
    // vectorSubtract(_nodes[i].adjacent_variables, _nodes[p].simple_variables);

    // element absorption
    vectorSubtract(_nodes[i].adjacent_elements,
                   _nodes[p].adjacent_elements); // \ \epsilon_p
    _nodes[i].adjacent_elements.insert(
        std::upper_bound(_nodes[i].adjacent_elements.begin(),
                         _nodes[i].adjacent_elements.end(), p),
        p); // \cup p

    // update degree and reinsert to principle map
    _nodes[i].degree = getDegree(i);
    auto it = _degree_to_principle.try_emplace(_nodes[i].degree,
                                               std::move(_cb_pool.acquire()));
    it.first->second->nonOverwritePush(i); // update node i's degree
  }
  return std::move(Lp);
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::supervariableMerge(
    const std::vector<COLTYPE> &fillins) {
  _hash_table.clear();
  for (size_t i = 0; i < fillins.size(); ++i) {
    auto it =
        _hash_table.try_emplace(hash(fillins[i]), std::move(_pool.acquire()));
    it.first->second->push_back(fillins[i]); // insert fillin to hash table
  }

  for (auto &it : _hash_table) {
    std::cout << "hash: " << it.first << std::endl;
    auto &vec = *(it.second);
    for (size_t i = 0; i < vec.size(); ++i) {
      const COLTYPE node = vec[i];
      if (node == invalid) {
        // skip merged
        continue;
      }
      auto degree = _nodes[node].degree;
      for (size_t j = i + 1; j < vec.size(); ++j) {
        if (isDistinguishable(node, vec[j])) {
          continue; // distinguishable
        }
        merge(node, vec[j]);
        vec[j] = invalid; // mark as merged
      }

      // if node degree is changed, reinsert to principle map
      if (_nodes[node].degree != degree) {
        std::cout << "update degree " << _nodes[node].degree << std::endl;
        auto it2 = _degree_to_principle.try_emplace(
            _nodes[node].degree, std::move(_cb_pool.acquire()));
        it2.first->second->nonOverwritePush(node);
      }
    }
  }
}

template <typename COLTYPE, MinimumDegree MD>
COLTYPE QuotientGraph<COLTYPE, MD>::getExternalDegree(const COLTYPE p) {
  auto temp = _pool.acquire();
  principleVector(_nodes[p].adjacent_variables, *temp);
  COLTYPE degree = vectorSubtractSize(_nodes[p].adjacent_variables,
                                      _nodes[p].simple_variables);

  auto &adj_elements = _nodes[p].adjacent_elements;
  __vectors.clear();
  COLTYPE pos = 0;
  for (size_t i = 0; i < adj_elements.size(); ++i) {
    COLTYPE j = adj_elements[i];
    if (_nodes[j].degree != element) {
      continue;
    }
    principleVector(_nodes[j].adjacent_variables, *temp);
    __vectors.push_back(&_nodes[j].adjacent_variables);
    adj_elements[pos++] = j; // keep only element nodes
  }
  adj_elements.resize(pos); // remove non-element nodes
  mergeKVectors(__vectors, *temp, std::optional<COLTYPE>(p));
  degree += vectorSubtractSize(*temp, _nodes[p].simple_variables);
  return degree;
}

template <typename COLTYPE, MinimumDegree MD>
COLTYPE QuotientGraph<COLTYPE, MD>::getExactDegree(const COLTYPE i) {
  auto temp1 = _pool.acquire();
  auto temp2 = _pool.acquire();
  getFillins(i, *temp1, *temp2);
  return static_cast<COLTYPE>(temp1->size());
}

template <typename COLTYPE, MinimumDegree MD>
COLTYPE QuotientGraph<COLTYPE, MD>::getApproximateDegree(const COLTYPE i) {
  auto temp1 = _pool.acquire();
  auto temp2 = _pool.acquire();
  getFillins(i, *temp1, *temp2);
  return static_cast<COLTYPE>(temp1->size());
}

template <typename COLTYPE, MinimumDegree MD>
COLTYPE QuotientGraph<COLTYPE, MD>::hash(const COLTYPE i) const {
  COLTYPE hash_value = 0;
  for (auto j : _nodes[i].adjacent_variables) {
    assert(j == _union_find.Find(j));
    hash_value += j;
    hash_value %= static_cast<COLTYPE>(_nodes.size());
  }
  for (auto j : _nodes[i].adjacent_elements) {
    assert(j == _union_find.Find(j));
    hash_value += j;
    hash_value %= static_cast<COLTYPE>(_nodes.size());
  }
  return hash_value;
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::principleVector(std::vector<COLTYPE> &vec,
                                                 std::vector<COLTYPE> &temp) {
  temp.clear();
  for (auto i : vec) {
    auto principleNode = _union_find.Find(i);
    if (_nodes[principleNode].degree != invalid) {
      temp.push_back(principleNode);
    }
  }
  std::sort(temp.begin(), temp.end());
  vec.clear();
  for (auto i : temp) {
    if (vec.empty() || vec.back() != i) {
      vec.push_back(i);
    }
  }
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::removeElementNode(const COLTYPE i) {
  assert(i == _union_find.Find(i));
  assert(_nodes[i].degree == element);
  _nodes[i].adjacent_variables.clear();
  _nodes[i].adjacent_variables.shrink_to_fit();
  _nodes[i].degree = invalid;
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::toElementNode(const COLTYPE i) {
  assert(i == _union_find.Find(i));
  _nodes[i].adjacent_elements.clear();
  _nodes[i].adjacent_elements.shrink_to_fit();
  _nodes[i].simple_variables.clear();
  _nodes[i].simple_variables.shrink_to_fit();
  _nodes[i].degree = element;
}

template <typename COLTYPE, MinimumDegree MD>
void QuotientGraph<COLTYPE, MD>::clearNode(const COLTYPE i) {
  assert(i == _union_find.Find(i));
  _nodes[i].adjacent_variables.clear();
  _nodes[i].adjacent_variables.shrink_to_fit();
  _nodes[i].adjacent_elements.clear();
  _nodes[i].adjacent_elements.shrink_to_fit();
  _nodes[i].simple_variables.clear();
  _nodes[i].simple_variables.shrink_to_fit();
  _nodes[i].degree = invalid;
}

// template <typename COLTYPE, MinimumDegree MD>
// void QuotientGraph<COLTYPE, MD>::updateWeight(const std::vector<COLTYPE> &Lp)
// {
//   for (auto i : Lp) {
//     assert(i == _union_find.Find(i));
//     for (auto j : _nodes[i].adjacent_elements) {
//       assert(j == _union_find.Find(j));
//       _nodes[j].weight =
//           static_cast<COLTYPE>(_nodes[j].adjacent_variables.size());
//     }
//   }
// }

template class QuotientGraph<int>;
template void QuotientGraph<int>::operator()(const int nnodes, int const *ai,
                                             int const *aj, int *perm,
                                             int *iperm);

template void QuotientGraph<int>::initialize(const int nnodes, int const *ai,
                                             int const *aj, int *&perm);

} // namespace reordering