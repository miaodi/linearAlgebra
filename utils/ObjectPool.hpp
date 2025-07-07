#pragma once

#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <iostream>

namespace utils {
// https://stackoverflow.com/questions/27827923/c-object-pool-that-provides-items-as-smart-pointers-that-are-returned-to-pool#comment44061893_27827923
template <typename T> class ObjectPool {
  class ObjectDeleter {
  public:
    ObjectDeleter(std::weak_ptr<std::vector<T *>> pool) : _poolPtr(pool) {}
    void operator()(T *obj) {
      if (auto poolPtr = _poolPtr.lock()) {
        poolPtr->push_back(obj);
      } else {
        std::default_delete<T>()(
            obj); // Fallback to default delete if pool is no longer available
      }
    }

  private:
    std::weak_ptr<std::vector<T *>> _poolPtr;
  };

public:
  using ptr_type = std::unique_ptr<T, ObjectDeleter>;
  using value_type = T;
  ObjectPool(size_t size = 0) : _pool(std::make_shared<std::vector<T *>>()) {
    // Reserve space in the pool and create objects
    if (size > 0) {
      _pool->reserve(size);
      for (size_t i = 0; i < size; ++i) {
        _pool->push_back(new T());
      }
    }
  }
  ~ObjectPool() {
    for (auto ptr : *_pool) {
      delete ptr;
    }
  }

  ptr_type acquire() {
    // std::cout<<"acquire from pool, current size: "<<_pool->size()<<std::endl;
    T *obj = nullptr;
    if (!_pool->empty()) {
      obj = _pool->back();
      _pool->pop_back();
    } else {
      obj = new T();
    }
    if (_objectPrep) {
      _objectPrep(
          obj); // Prepare the object if a preparation function is provided
    }
    // If no object is available, create a new one
    return ptr_type(obj, ObjectDeleter(_pool));
  }

  bool empty() const { return _pool->empty(); }

  size_t size() const { return _pool->size(); }

  void setObjectPrep(std::function<void(T *)> prepFunc) {
    _objectPrep = prepFunc;
  }

  // Prevent copy and assignment
  ObjectPool(const ObjectPool &) = delete;
  ObjectPool &operator=(const ObjectPool &) = delete;

private:
  std::shared_ptr<std::vector<T *>> _pool;
  std::function<void(T *)> _objectPrep;
};
} // namespace utils