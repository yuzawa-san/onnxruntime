package ai.onnxruntime;

import java.io.IOException;

/**
 * This is the base class for anything backed by JNI. It manages open versus closed state and hides
 * the underlying native handle.
 */
abstract class NativeObject implements AutoCloseable {

  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  private final long handle;

  private volatile boolean closed;

  NativeObject(long handle) {
    this.handle = handle;
    this.closed = false;
  }

  /**
   * Generates a description with the backing native object's handle.
   *
   * @return the description
   */
  @Override
  public String toString() {
    return getClass().getSimpleName() + "@" + Long.toHexString(handle);
  }

  /**
   * Check if the resource is closed.
   *
   * @return true if closed
   */
  public final boolean isClosed() {
    return closed;
  }

  /**
   * This internal method allows implementations to specify the manner in which this object's
   * backing native object(s) are released/freed/closed.
   *
   * @param handle a long representation of the address of the backing native object.
   */
  abstract void doClose(long handle);

  /**
   * Releases any native resources related to this object.
   *
   * <p>This method must be called or else the application will leak off-heap memory. It is best
   * practice to use a try-with-resources which will ensure this method always be called. This
   * method will block until any active usages are complete.
   */
  @Override
  public void close() {
    if (closed) {
      return;
    }
    doClose(handle);
    closed = true;
  }

  /**
   * Get a usage to the backing native object. This method ensures the object is open. It is
   * recommended this be used with a try-with-resources to ensure the {@link NativeUsage} is closed
   * and does not leak out of scope. The usage should not be shared between threads.
   *
   * @return a usage from which the backing native object's handle can be used.
   */
  final NativeUsage use() {
    return new NativeUsage();
  }

  /** A managed usage to the backing native object. */
  final class NativeUsage implements AutoCloseable {

    public NativeUsage() {
      if (isClosed()) {
        throw new IllegalStateException(
            NativeObject.this.getClass().getSimpleName() + " has been closed already.");
      }
    }

    public long handle() {
      return handle;
    }

    @Override
    public void close() {
      // pass
    }
  }
}
