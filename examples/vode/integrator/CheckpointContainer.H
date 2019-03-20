#ifndef _CHECKPOINT_CONTAINER_H
#define _CHECKPOINT_CONTAINER_H

// Restriction: EnumClass elements must be convertible to indexes!!!
// That is, they must range from 0 to N-1, for N EnumClass elements.

template<class EnumClass, size_t N> class CheckpointContainer {
private:    
    bool checkpoint[N];

    __host__ __device__
    bool* begin() {
        return checkpoint;
    }

    __host__ __device__
    bool* end() {
        return checkpoint + N;
    }

    __host__ __device__
    bool& operator[] (unsigned int i) {
        return *(checkpoint + i);
    }
public:
    CheckpointContainer() {
        unset_all_checkpoints();
    }

    __host__ __device__
    unset_all() {
        for (auto& x : *this) {
            x = false;
        }
    }

    __host__ __device__
    set(EnumClass checkpoint_id) {
        unsigned int i = static_cast<unsigned int>(checkpoint_id);
        *this[i] = true;
    }

    __host__ __device__
    unset(EnumClass checkpoint_id) {
        unsigned int i = static_cast<unsigned int>(checkpoint_id);
        *this[i] = false;
    }

    __host__ __device__
    bool not_done(EnumClass checkpoint_id) {
        unsigned int i = static_cast<unsigned int>(checkpoint_id);        
        return !(*this[i]);
    }

    __host__ __device__
    bool is_done(EnumClass checkpoint_id) {
        unsigned int i = static_cast<unsigned int>(checkpoint_id);        
        return (*this[i]);
    }    
};
#endif
