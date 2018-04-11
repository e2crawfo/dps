set -ev
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
echo "$TF_CFLAGS"
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
echo "$TF_LFLAGS"
g++ -std=c++11 -shared ops/render_sprites_ops.cc kernels/render_sprites_ops.cc -o _render_sprites.so -fPIC -Ikernels ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2