/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

/**
 * A Java object wrapping an OnnxTensor. Tensors are the main input to the library, and can also be
 * returned as outputs.
 */
public class OnnxTensor extends NativeObject implements OnnxValue {

  private final OrtAllocator allocator;

  private final TensorInfo info;

  /**
   * This reference is held for OnnxTensors backed by a Java nio buffer to ensure the buffer does
   * not go out of scope while the OnnxTensor exists.
   */
  private final Buffer buffer;

  OnnxTensor(long nativeHandle, OrtAllocator allocator, TensorInfo info) {
    this(nativeHandle, allocator, info, null);
  }

  OnnxTensor(long nativeHandle, OrtAllocator allocator, TensorInfo info, Buffer buffer) {
    super(nativeHandle);
    this.allocator = allocator;
    this.info = info;
    this.buffer = buffer;
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_TENSOR;
  }

  /**
   * Either returns a boxed primitive if the Tensor is a scalar, or a multidimensional array of
   * primitives if it has multiple dimensions.
   *
   * <p>Java multidimensional arrays are quite slow for more than 2 dimensions, in that case it is
   * recommended you use the java.nio.Buffer extractors below (e.g. {@link #getFloatBuffer}).
   *
   * @return A Java value.
   * @throws OrtException If the value could not be extracted as the Tensor is invalid, or if the
   *     native code encountered an error.
   */
  @Override
  public Object getValue() throws OrtException {
    try (NativeUsage tensorReference = use();
        NativeUsage allocatorReference = allocator.use()) {
      if (info.isScalar()) {
        switch (info.type) {
          case FLOAT:
            return getFloat(
                OnnxRuntime.ortApiHandle, tensorReference.handle(), info.onnxType.value);
          case DOUBLE:
            return getDouble(OnnxRuntime.ortApiHandle, tensorReference.handle());
          case INT8:
            return getByte(OnnxRuntime.ortApiHandle, tensorReference.handle(), info.onnxType.value);
          case INT16:
            return getShort(
                OnnxRuntime.ortApiHandle, tensorReference.handle(), info.onnxType.value);
          case INT32:
            return getInt(OnnxRuntime.ortApiHandle, tensorReference.handle(), info.onnxType.value);
          case INT64:
            return getLong(OnnxRuntime.ortApiHandle, tensorReference.handle(), info.onnxType.value);
          case BOOL:
            return getBool(OnnxRuntime.ortApiHandle, tensorReference.handle());
          case STRING:
            return getString(
                OnnxRuntime.ortApiHandle, tensorReference.handle(), allocatorReference.handle());
          case UNKNOWN:
          default:
            throw new OrtException("Extracting the value of an invalid Tensor.");
        }
      } else {
        Object carrier = info.makeCarrier();
        getArray(
            OnnxRuntime.ortApiHandle,
            tensorReference.handle(),
            allocatorReference.handle(),
            carrier);
        return carrier;
      }
    }
  }

  @Override
  public TensorInfo getInfo() {
    return info;
  }

  @Override
  public String toString() {
    return super.toString() + "(info=" + info.toString() + ")";
  }

  @Override
  protected void doClose(long handle) {
    close(OnnxRuntime.ortApiHandle, handle);
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a ByteBuffer.
   *
   * <p>This method returns null if the OnnxTensor contains Strings as they are stored externally to
   * the OnnxTensor.
   *
   * @return A ByteBuffer copy of the OnnxTensor.
   */
  public ByteBuffer getByteBuffer() {
    if (info.type == OnnxJavaType.STRING) {
      return null;
    }
    try (NativeUsage tensorReference = use()) {
      ByteBuffer buffer = getBuffer(OnnxRuntime.ortApiHandle, tensorReference.handle());
      ByteBuffer output = ByteBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a FloatBuffer if it can be losslessly converted
   * into a float (i.e. it's a float or fp16), otherwise it returns null.
   *
   * @return A FloatBuffer copy of the OnnxTensor.
   */
  public FloatBuffer getFloatBuffer() {
    if (info.type == OnnxJavaType.FLOAT) {
      if (info.onnxType == TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        // if it's fp16 we need to copy it out by hand.
        ShortBuffer buffer = getBuffer().asShortBuffer();
        int bufferCap = buffer.capacity();
        FloatBuffer output = FloatBuffer.allocate(bufferCap);
        for (int i = 0; i < bufferCap; i++) {
          output.put(fp16ToFloat(buffer.get(i)));
        }
        output.rewind();
        return output;
      } else {
        // if it's fp32 use the efficient copy.
        FloatBuffer buffer = getBuffer().asFloatBuffer();
        FloatBuffer output = FloatBuffer.allocate(buffer.capacity());
        output.put(buffer);
        output.rewind();
        return output;
      }
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a DoubleBuffer if the underlying type is a
   * double, otherwise it returns null.
   *
   * @return A DoubleBuffer copy of the OnnxTensor.
   */
  public DoubleBuffer getDoubleBuffer() {
    if (info.type == OnnxJavaType.DOUBLE) {
      DoubleBuffer buffer = getBuffer().asDoubleBuffer();
      DoubleBuffer output = DoubleBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a ShortBuffer if the underlying type is int16 or
   * uint16, otherwise it returns null.
   *
   * @return A ShortBuffer copy of the OnnxTensor.
   */
  public ShortBuffer getShortBuffer() {
    if (info.type == OnnxJavaType.INT16) {
      ShortBuffer buffer = getBuffer().asShortBuffer();
      ShortBuffer output = ShortBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as an IntBuffer if the underlying type is int32 or
   * uint32, otherwise it returns null.
   *
   * @return An IntBuffer copy of the OnnxTensor.
   */
  public IntBuffer getIntBuffer() {
    if (info.type == OnnxJavaType.INT32) {
      IntBuffer buffer = getBuffer().asIntBuffer();
      IntBuffer output = IntBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Returns a copy of the underlying OnnxTensor as a LongBuffer if the underlying type is int64 or
   * uint64, otherwise it returns null.
   *
   * @return A LongBuffer copy of the OnnxTensor.
   */
  public LongBuffer getLongBuffer() {
    if (info.type == OnnxJavaType.INT64) {
      LongBuffer buffer = getBuffer().asLongBuffer();
      LongBuffer output = LongBuffer.allocate(buffer.capacity());
      output.put(buffer);
      output.rewind();
      return output;
    } else {
      return null;
    }
  }

  /**
   * Wraps the OrtTensor pointer in a direct byte buffer of the native platform endian-ness. Unless
   * you really know what you're doing, you want this one rather than the native call {@link
   * OnnxTensor#getBuffer(long,long)}.
   *
   * @return A ByteBuffer wrapping the data.
   */
  private ByteBuffer getBuffer() {
    try (NativeUsage tensorReference = use()) {
      return getBuffer(OnnxRuntime.ortApiHandle, tensorReference.handle())
          .order(ByteOrder.nativeOrder());
    }
  }

  /**
   * Wraps the OrtTensor pointer in a direct byte buffer.
   *
   * @param apiHandle The OrtApi pointer.
   * @param nativeHandle The OrtTensor pointer.
   * @return A ByteBuffer wrapping the data.
   */
  private native ByteBuffer getBuffer(long apiHandle, long nativeHandle);

  private native float getFloat(long apiHandle, long nativeHandle, int onnxType)
      throws OrtException;

  private native double getDouble(long apiHandle, long nativeHandle) throws OrtException;

  private native byte getByte(long apiHandle, long nativeHandle, int onnxType) throws OrtException;

  private native short getShort(long apiHandle, long nativeHandle, int onnxType)
      throws OrtException;

  private native int getInt(long apiHandle, long nativeHandle, int onnxType) throws OrtException;

  private native long getLong(long apiHandle, long nativeHandle, int onnxType) throws OrtException;

  private native String getString(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native boolean getBool(long apiHandle, long nativeHandle) throws OrtException;

  private native void getArray(
      long apiHandle, long nativeHandle, long allocatorHandle, Object carrier) throws OrtException;

  private native void close(long apiHandle, long nativeHandle);

  /**
   * Mirrors the conversion in the C code. It's not precise if there are subnormal values, nor does
   * it preserve all the different kinds of NaNs (which aren't representable in Java anyway).
   *
   * @param input A uint16_t representing an IEEE half precision float.
   * @return A float.
   */
  private static float fp16ToFloat(short input) {
    int output =
        ((input & 0x8000) << 16) | (((input & 0x7c00) + 0x1C000) << 13) | ((input & 0x03FF) << 13);
    return Float.intBitsToFloat(output);
  }

  /**
   * Create a Tensor from a Java primitive or String multidimensional array. The shape is inferred
   * from the array using reflection. The default allocator is used.
   *
   * @param env The current OrtEnvironment.
   * @param data The data to store in a tensor.
   * @return An OnnxTensor storing the data.
   * @throws OrtException If the onnx runtime threw an error.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, Object data) throws OrtException {
    return createTensor(env, OrtAllocator.DEFAULT_ALLOCATOR, data);
  }

  /**
   * Create a Tensor from a Java primitive or String multidimensional array. The shape is inferred
   * from the array using reflection.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The data to store in a tensor.
   * @return An OnnxTensor storing the data.
   * @throws OrtException If the onnx runtime threw an error.
   */
  static OnnxTensor createTensor(OrtEnvironment env, OrtAllocator allocator, Object data)
      throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage allocatorReference = allocator.use()) {
      long allocatorHandle = allocatorReference.handle();
      TensorInfo info = TensorInfo.constructFromJavaArray(data);
      if (info.type == OnnxJavaType.STRING) {
        if (info.shape.length == 0) {
          return new OnnxTensor(
              createString(OnnxRuntime.ortApiHandle, allocatorHandle, (String) data),
              allocator,
              info);
        } else {
          return new OnnxTensor(
              createStringTensor(
                  OnnxRuntime.ortApiHandle,
                  allocatorHandle,
                  OrtUtil.flattenString(data),
                  info.shape),
              allocator,
              info);
        }
      } else {
        if (info.shape.length == 0) {
          data = OrtUtil.convertBoxedPrimitiveToArray(data);
        }
        return new OnnxTensor(
            createTensor(
                OnnxRuntime.ortApiHandle, allocatorHandle, data, info.shape, info.onnxType.value),
            allocator,
            info);
      }
    }
  }

  /**
   * Create a tensor from a flattened string array.
   *
   * <p>Requires the array to be flattened in row-major order. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data
   * @param shape the shape of the tensor
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, String[] data, long[] shape)
      throws OrtException {
    return createTensor(env, OrtAllocator.DEFAULT_ALLOCATOR, data, shape);
  }

  /**
   * Create a tensor from a flattened string array.
   *
   * <p>Requires the array to be flattened in row-major order.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data
   * @param shape the shape of the tensor
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, String[] data, long[] shape) throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage allocatorReference = allocator.use()) {
      TensorInfo info =
          new TensorInfo(
              shape,
              OnnxJavaType.STRING,
              TensorInfo.OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
      return new OnnxTensor(
          createStringTensor(OnnxRuntime.ortApiHandle, allocatorReference.handle(), data, shape),
          allocator,
          info);
    }
  }

  /**
   * Create an OnnxTensor backed by a direct FloatBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, FloatBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, OrtAllocator.DEFAULT_ALLOCATOR, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct FloatBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, FloatBuffer data, long[] shape)
      throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage allocatorReference = allocator.use()) {
      OnnxJavaType type = OnnxJavaType.FLOAT;
      int bufferSize = data.capacity() * type.size;
      FloatBuffer tmp;
      if (data.isDirect()) {
        tmp = data;
      } else {
        ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        tmp = buffer.asFloatBuffer();
        tmp.put(data);
      }
      TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
      return new OnnxTensor(
          createTensorFromBuffer(
              OnnxRuntime.ortApiHandle,
              allocatorReference.handle(),
              tmp,
              bufferSize,
              shape,
              info.onnxType.value),
          allocator,
          info,
          tmp);
    }
  }

  /**
   * Create an OnnxTensor backed by a direct DoubleBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, DoubleBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, OrtAllocator.DEFAULT_ALLOCATOR, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct DoubleBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, DoubleBuffer data, long[] shape)
      throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage allocatorReference = allocator.use()) {
      OnnxJavaType type = OnnxJavaType.DOUBLE;
      int bufferSize = data.capacity() * type.size;
      DoubleBuffer tmp;
      if (data.isDirect()) {
        tmp = data;
      } else {
        ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        tmp = buffer.asDoubleBuffer();
        tmp.put(data);
      }
      TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
      return new OnnxTensor(
          createTensorFromBuffer(
              OnnxRuntime.ortApiHandle,
              allocatorReference.handle(),
              tmp,
              bufferSize,
              shape,
              info.onnxType.value),
          allocator,
          info,
          tmp);
    }
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator. Tells the runtime it's {@link OnnxJavaType#INT8}.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, ByteBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, OrtAllocator.DEFAULT_ALLOCATOR, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Tells the runtime it's {@link OnnxJavaType#INT8}.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, ByteBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, allocator, data, shape, OnnxJavaType.INT8);
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator. Tells the runtime it's the specified type.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @param type The type to use for the byte buffer elements.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(
      OrtEnvironment env, ByteBuffer data, long[] shape, OnnxJavaType type) throws OrtException {
    return createTensor(env, OrtAllocator.DEFAULT_ALLOCATOR, data, shape, type);
  }

  /**
   * Create an OnnxTensor backed by a direct ByteBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Tells the runtime it's the specified type.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @param type The type to use for the byte buffer elements.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, ByteBuffer data, long[] shape, OnnxJavaType type)
      throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage allocatorReference = allocator.use()) {
      int bufferSize = data.capacity();
      ByteBuffer tmp;
      if (data.isDirect()) {
        tmp = data;
      } else {
        tmp = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        tmp.put(data);
      }
      TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
      return new OnnxTensor(
          createTensorFromBuffer(
              OnnxRuntime.ortApiHandle,
              allocatorReference.handle(),
              tmp,
              bufferSize,
              shape,
              info.onnxType.value),
          allocator,
          info,
          tmp);
    }
  }

  /**
   * Create an OnnxTensor backed by a direct ShortBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, ShortBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, OrtAllocator.DEFAULT_ALLOCATOR, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct ShortBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, ShortBuffer data, long[] shape)
      throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage allocatorReference = allocator.use()) {
      OnnxJavaType type = OnnxJavaType.INT16;
      int bufferSize = data.capacity() * type.size;
      ShortBuffer tmp;
      if (data.isDirect()) {
        tmp = data;
      } else {
        ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        tmp = buffer.asShortBuffer();
        tmp.put(data);
      }
      TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
      return new OnnxTensor(
          createTensorFromBuffer(
              OnnxRuntime.ortApiHandle,
              allocatorReference.handle(),
              tmp,
              bufferSize,
              shape,
              info.onnxType.value),
          allocator,
          info,
          tmp);
    }
  }

  /**
   * Create an OnnxTensor backed by a direct IntBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, IntBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, OrtAllocator.DEFAULT_ALLOCATOR, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct IntBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, IntBuffer data, long[] shape)
      throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage allocatorReference = allocator.use()) {
      OnnxJavaType type = OnnxJavaType.INT32;
      int bufferSize = data.capacity() * type.size;
      IntBuffer tmp;
      if (data.isDirect()) {
        tmp = data;
      } else {
        ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        tmp = buffer.asIntBuffer();
        tmp.put(data);
      }
      TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
      return new OnnxTensor(
          createTensorFromBuffer(
              OnnxRuntime.ortApiHandle,
              allocatorReference.handle(),
              tmp,
              bufferSize,
              shape,
              info.onnxType.value),
          allocator,
          info,
          tmp);
    }
  }

  /**
   * Create an OnnxTensor backed by a direct LongBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the default allocator.
   *
   * @param env The current OrtEnvironment.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  public static OnnxTensor createTensor(OrtEnvironment env, LongBuffer data, long[] shape)
      throws OrtException {
    return createTensor(env, OrtAllocator.DEFAULT_ALLOCATOR, data, shape);
  }

  /**
   * Create an OnnxTensor backed by a direct LongBuffer. The buffer should be in nativeOrder.
   *
   * <p>If the supplied buffer is not a direct buffer, a direct copy is created tied to the lifetime
   * of the tensor. Uses the supplied allocator.
   *
   * @param env The current OrtEnvironment.
   * @param allocator The allocator to use.
   * @param data The tensor data.
   * @param shape The shape of tensor.
   * @return An OnnxTensor of the required shape.
   * @throws OrtException Thrown if there is an onnx error or if the data and shape don't match.
   */
  static OnnxTensor createTensor(
      OrtEnvironment env, OrtAllocator allocator, LongBuffer data, long[] shape)
      throws OrtException {
    try (NativeUsage environmentReference = env.use();
        NativeUsage allocatorReference = allocator.use()) {
      OnnxJavaType type = OnnxJavaType.INT64;
      int bufferSize = data.capacity() * type.size;
      LongBuffer tmp;
      if (data.isDirect()) {
        tmp = data;
      } else {
        ByteBuffer buffer = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        tmp = buffer.asLongBuffer();
        tmp.put(data);
      }
      TensorInfo info = TensorInfo.constructFromBuffer(tmp, shape, type);
      return new OnnxTensor(
          createTensorFromBuffer(
              OnnxRuntime.ortApiHandle,
              allocatorReference.handle(),
              tmp,
              bufferSize,
              shape,
              info.onnxType.value),
          allocator,
          info,
          tmp);
    }
  }

  private static native long createTensor(
      long apiHandle, long allocatorHandle, Object data, long[] shape, int onnxType)
      throws OrtException;

  private static native long createTensorFromBuffer(
      long apiHandle,
      long allocatorHandle,
      Buffer data,
      long bufferSize,
      long[] shape,
      int onnxType)
      throws OrtException;

  private static native long createString(long apiHandle, long allocatorHandle, String data)
      throws OrtException;

  private static native long createStringTensor(
      long apiHandle, long allocatorHandle, Object[] data, long[] shape) throws OrtException;
}
