from std.collections import List
from std.ffi import OwnedDLHandle, c_int, c_ulong_long


def main() raises:
    var library = OwnedDLHandle("mojo/ffi/target/release/libdodge_mojo_ffi.so")
    var new_batch = library.get_function[c_ulong_long]("mojo_batch_new")
    var reset = library.get_function[c_int]("mojo_batch_reset")
    var step = library.get_function[c_int]("mojo_batch_step")
    var copy_observations = library.get_function[c_int](
        "mojo_batch_copy_observations"
    )
    var copy_positions = library.get_function[c_int](
        "mojo_batch_copy_positions"
    )
    var free_batch = library.get_function[NoneType]("mojo_batch_free")

    var seeds: List[UInt32] = [30181, 30169]
    var handle = new_batch(2, 4, 32, 0)
    if handle == c_ulong_long(0):
        print("failed to create batch")
        return
    var status = reset(handle, seeds.unsafe_ptr(), 2)
    if status != 0:
        print("reset failed")
        free_batch(handle)
        return
    var output = List[Float32](length=2 * 225, fill=0.0)
    status = copy_observations(handle, output.unsafe_ptr(), 2 * 225)
    if status != 0:
        print("copy failed")
        free_batch(handle)
        return
    var positions = List[Float32](length=2 * 2, fill=0.0)
    status = copy_positions(handle, positions.unsafe_ptr(), 2 * 2)
    if status != 0:
        print("position copy failed")
        free_batch(handle)
        return
    var checksum = Float64(0.0)
    for index in range(len(output)):
        checksum += Float64(output[index]) * Float64(index + 1)
    print(
        "observation[0]=",
        output[0],
        "observation[1]=",
        output[1],
        "observation[224]=",
        output[224],
        "observation[225]=",
        output[225],
        "position0=",
        positions[0],
        positions[1],
        "position1=",
        positions[2],
        positions[3],
        "checksum=",
        checksum,
    )
    var actions: List[UInt8] = [8, 1]
    status = step(handle, actions.unsafe_ptr(), 2)
    if status != 0:
        print("step failed")
        free_batch(handle)
        return
    status = copy_observations(handle, output.unsafe_ptr(), 2 * 225)
    status = copy_positions(handle, positions.unsafe_ptr(), 2 * 2)
    checksum = Float64(0.0)
    for index in range(len(output)):
        checksum += Float64(output[index]) * Float64(index + 1)
    print(
        "after_step position0=",
        positions[0],
        positions[1],
        "position1=",
        positions[2],
        positions[3],
        "checksum=",
        checksum,
    )
    free_batch(handle)
