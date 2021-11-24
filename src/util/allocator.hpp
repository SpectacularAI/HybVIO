#ifndef DAZZLING_IMAGE_ALLOCATOR
#define DAZZLING_IMAGE_ALLOCATOR

#include "logging.hpp"
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

namespace util {
/**
 * A buffer and custom allocation mechanism for maintaining a buffer of
 * re-usable shared_ptr slots. Designed for image and texture storage in mind
 *  - enable pre-allocation of memory for a certain fixed capacity
 *  - lazy initialization to automatically determine the capacity
 *  - if the initial cacacity is exceeded, allocates more memory and re-uses
 *  - reference counting using std::shared_ptr to check which slots are free
 */
template <class Img> class Allocator {
private:
    static constexpr std::size_t DEFAULT_CAPACITY_INCREASE = 5;
    const std::size_t capacityIncrease;
    const std::size_t maxCapacity;

    const std::function< std::unique_ptr<Img>() > allocator;
    std::vector< std::shared_ptr<Img> > buf;
    std::size_t counter = 0;

    void increaseCapacity() {
        const auto n0 = buf.size();
        while (buf.size() < n0 + capacityIncrease) {
            buf.emplace_back(allocator());
        }

        // "safety valve". If more than this many elements were simultaneously
        // allocated, something has gone wrong
        assert(maxCapacity == 0 || buf.size() <= maxCapacity);
        log_debug("increased Allocator capacity %zu -> %zu", n0, buf.size());
    }

public:
    Allocator(const std::function< std::unique_ptr<Img>() > &allocator,
        std::size_t initialCapacity = 0,
        std::size_t capacityIncrease = DEFAULT_CAPACITY_INCREASE,
        std::size_t maxCapacity = DEFAULT_CAPACITY_INCREASE * 20)
    :
        capacityIncrease(capacityIncrease),
        maxCapacity(maxCapacity),
        allocator(allocator)
    {
        assert(allocator);
        while (buf.size() < initialCapacity) buf.emplace_back(allocator());
    }

    std::shared_ptr<Img> next() {
        if (buf.empty()) increaseCapacity();
        for (std::size_t failures = 0; failures < buf.size(); failures++) {
            // note: this part should be OK without extra syncronization as
            // each thread that may use the object referred to by this ptr
            // does that through its own std::shared_ptr. I.e., we don't use
            // access the same "shared_ptr&", from multiple threads.
            auto &shared = buf.at((counter = (counter+1) % buf.size()));
            if (shared.unique()) return shared;
        }
        increaseCapacity();
        return next();
    }
};

}

#endif
