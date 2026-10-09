import CryptoKit
import Foundation
import zlib

/// Tensor samples are original little-endian storage codes, without numeric
/// casts. A strided tensor is gathered into row-major storage byte-for-byte.
struct NativeTensor: Sendable {
    let dtype: String
    let shape: [Int]
    let bytes: Data
}
struct NativeTorchWeights: Sendable {
    let sha256: String
    let tensors: [String: NativeTensor]
    let metadata: [String: String]
}

/// Reads modern torch.save ZIP state dictionaries as data. Pickle opcodes are
/// interpreted by a closed parser; no imported module, constructor, or code
/// from the file is executed. Legacy non-ZIP and object checkpoints are refused.
enum NativeTorchCheckpoint {
    static func load(url: URL, expectedSHA256: String? = nil) throws -> NativeTorchWeights {
        try load(bytes: Data(contentsOf: url), expectedSHA256: expectedSHA256)
    }
    static func load(bytes: Data, expectedSHA256: String? = nil) throws -> NativeTorchWeights {
        try Task.checkCancellation()
        let sha256 = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        guard expectedSHA256 == nil || expectedSHA256 == sha256 else { throw failure("selected weights changed") }
        let archive = try TorchZIP(bytes)
        let picklePaths = archive.entries.keys.filter { $0.hasSuffix("/data.pkl") }
        guard picklePaths.count == 1, let picklePath = picklePaths.first else { throw failure("expected one tensor state dictionary") }
        let prefix = String(picklePath.dropLast("data.pkl".count))
        for path in archive.entries.keys {
            guard path.hasPrefix(prefix) else { throw failure("ZIP contains unrelated archive roots") }
        }
        if archive.entries[prefix + "byteorder"] != nil {
            guard try String(decoding: archive.read(prefix + "byteorder"), as: UTF8.self) == "little" else { throw failure("only little-endian tensor storage is supported") }
        }
        let pickle = try archive.read(picklePath)
        guard pickle.count <= 100_000_000 else { throw failure("tensor metadata is too large") }
        let state = try TorchPickle(pickle).decode()
        guard case .dictionary(let pairs) = state.kind, !pairs.isEmpty else { throw failure("checkpoint must be a nonempty tensor state dictionary") }
        var tensors: [String: NativeTensor] = [:]
        var storages: [String: Data] = [:]
        for (key, value) in pairs {
            try Task.checkCancellation()
            guard case .string(let name) = key.kind, !name.isEmpty, tensors[name] == nil,
                  case .tensor(let tensor) = value.kind else { throw failure("state dictionary contains an object or duplicate tensor name") }
            let storageKey = tensor.storage.key
            let storage: Data
            if let cached = storages[storageKey] { storage = cached }
            else {
                storage = try archive.read(prefix + "data/" + storageKey)
                storages[storageKey] = storage
            }
            let dtype = tensor.dtype ?? tensor.storage.dtype
            guard let width = widths[dtype] else { throw failure("unsupported tensor storage dtype") }
            let storageCount = try multiplied(tensor.storage.count, tensor.storage.dtype == "UNTYPED" ? 1 : widths[tensor.storage.dtype] ?? 0)
            guard storage.count == storageCount else { throw failure("tensor storage size differs from metadata") }
            tensors[name] = NativeTensor(dtype: dtype, shape: tensor.shape, bytes: try tensorBytes(tensor, width: width, storage: storage))
        }
        return NativeTorchWeights(sha256: sha256, tensors: tensors, metadata: ["format": "torch-save-zip-state-dictionary", "byteorder": "little"])
    }

    fileprivate static let widths = ["BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "I32": 4,
        "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8]
    fileprivate static func failure(_ message: String) -> NativeCheckpointError { .invalid("PyTorch archive: " + message) }
    fileprivate static func multiplied(_ a: Int, _ b: Int) throws -> Int {
        let result = a.multipliedReportingOverflow(by: b)
        guard a >= 0, b >= 0, !result.overflow else { throw failure("tensor dimensions overflow") }
        return result.partialValue
    }
    fileprivate static func added(_ a: Int, _ b: Int) throws -> Int {
        let result = a.addingReportingOverflow(b)
        guard a >= 0, b >= 0, !result.overflow else { throw failure("tensor offsets overflow") }
        return result.partialValue
    }
    private static func tensorBytes(_ tensor: TorchTensor, width: Int, storage: Data) throws -> Data {
        guard tensor.shape.count == tensor.stride.count, tensor.shape.count <= 32,
              tensor.shape.allSatisfy({ $0 >= 0 }), tensor.stride.allSatisfy({ $0 >= 0 }), tensor.offset >= 0 else { throw failure("invalid tensor shape or stride") }
        var count = 1, maximum = tensor.offset
        for (dimension, stride) in zip(tensor.shape, tensor.stride) {
            count = try multiplied(count, dimension)
            if dimension > 0 { maximum = try added(maximum, multiplied(dimension - 1, stride)) }
        }
        let byteCount = try multiplied(count, width)
        guard byteCount <= storage.count else { throw failure("tensor view exceeds storage") }
        if count == 0 {
            guard try multiplied(tensor.offset, width) <= storage.count else { throw failure("empty tensor offset exceeds storage") }
            return Data()
        }
        guard try multiplied(added(maximum, 1), width) <= storage.count else { throw failure("tensor view exceeds storage") }
        var expectedStride = 1, contiguous = true
        for axis in tensor.shape.indices.reversed() {
            if tensor.shape[axis] > 1, tensor.stride[axis] != expectedStride { contiguous = false }
            expectedStride = try multiplied(expectedStride, tensor.shape[axis])
        }
        if contiguous {
            let start = try multiplied(tensor.offset, width)
            if start == 0, byteCount == storage.count { return storage }
            return storage.subdata(in: start..<start + byteCount)
        }
        var output = Data(count: byteCount)
        try output.withUnsafeMutableBytes { destination in
            try storage.withUnsafeBytes { source in
                for linear in 0..<count {
                    if linear & 0x3ffff == 0 { try Task.checkCancellation() }
                    var remaining = linear, index = tensor.offset
                    for axis in tensor.shape.indices.reversed() {
                        let coordinate = remaining % tensor.shape[axis]
                        remaining /= tensor.shape[axis]
                        index += coordinate * tensor.stride[axis]
                    }
                    destination.baseAddress!.advanced(by: linear * width).copyMemory(from: source.baseAddress!.advanced(by: index * width), byteCount: width)
                }
            }
        }
        return output
    }
}

private struct TorchZIP {
    struct Entry { let name: String; let method: UInt16; let crc: UInt32; let compressed: Int; let size: Int; let offset: Int }
    let bytes: Data
    let entries: [String: Entry]
    init(_ bytes: Data) throws {
        self.bytes = bytes
        guard bytes.count >= 22 else { throw NativeTorchCheckpoint.failure("not a ZIP archive") }
        var end: Int?
        for offset in stride(from: bytes.count - 22, through: max(0, bytes.count - 65557), by: -1) {
            if bytes.u32(offset) == 0x06054b50, offset + 22 + Int(bytes.u16(offset + 20)) == bytes.count { end = offset; break }
        }
        guard let end, bytes.u16(end + 4) == 0, bytes.u16(end + 6) == 0,
              bytes.u16(end + 8) == bytes.u16(end + 10) else { throw NativeTorchCheckpoint.failure("invalid ZIP directory") }
        var count = Int(bytes.u16(end + 10)), size = Int(bytes.u32(end + 12)), start = Int(bytes.u32(end + 16))
        if count == 65535 || size == Int(UInt32.max) || start == Int(UInt32.max) {
            guard end >= 20, bytes.u32(end - 20) == 0x07064b50, bytes.u32(end - 16) == 0,
                  bytes.u32(end - 4) == 1, let record = Int(exactly: bytes.u64(end - 12)),
                  record <= bytes.count - 56, bytes.u32(record) == 0x06064b50,
                  bytes.u32(record + 16) == 0, bytes.u32(record + 20) == 0,
                  bytes.u64(record + 24) == bytes.u64(record + 32),
                  let total = Int(exactly: bytes.u64(record + 32)), let directorySize = Int(exactly: bytes.u64(record + 40)),
                  let directoryStart = Int(exactly: bytes.u64(record + 48)) else { throw NativeTorchCheckpoint.failure("invalid ZIP64 directory") }
            count = total; size = directorySize; start = directoryStart
        }
        guard count > 0, count <= 100_000, start >= 0, size >= 0, start <= end, size <= end - start else { throw NativeTorchCheckpoint.failure("ZIP directory exceeds file bounds") }
        var entries: [String: Entry] = [:], cursor = start
        for _ in 0..<count {
            guard cursor <= start + size - 46, bytes.u32(cursor) == 0x02014b50 else { throw NativeTorchCheckpoint.failure("invalid ZIP member descriptor") }
            let flags = bytes.u16(cursor + 8), method = bytes.u16(cursor + 10)
            guard flags & 1 == 0, [UInt16(0), 8].contains(method), bytes.u16(cursor + 34) == 0 else { throw NativeTorchCheckpoint.failure("encrypted or unsupported ZIP member") }
            let nameSize = Int(bytes.u16(cursor + 28)), extraSize = Int(bytes.u16(cursor + 30)), commentSize = Int(bytes.u16(cursor + 32))
            let recordSize = 46 + nameSize + extraSize + commentSize
            guard recordSize <= start + size - cursor,
                  let name = String(data: bytes.subdata(in: cursor + 46..<cursor + 46 + nameSize), encoding: .utf8),
                  !name.isEmpty, !name.hasPrefix("/"), !name.split(separator: "/").contains(".."), !name.contains("\\"), entries[name] == nil else { throw NativeTorchCheckpoint.failure("unsafe or duplicate ZIP name") }
            var compressed = UInt64(bytes.u32(cursor + 20)), expanded = UInt64(bytes.u32(cursor + 24)), offset = UInt64(bytes.u32(cursor + 42))
            if compressed == UInt32.max || expanded == UInt32.max || offset == UInt32.max {
                var extraCursor = cursor + 46 + nameSize, found = false
                let extraEnd = extraCursor + extraSize
                while extraCursor + 4 <= extraEnd {
                    let tag = bytes.u16(extraCursor), length = Int(bytes.u16(extraCursor + 2)); extraCursor += 4
                    guard length <= extraEnd - extraCursor else { throw NativeTorchCheckpoint.failure("truncated ZIP extra field") }
                    if tag == 1 {
                        var valueCursor = extraCursor
                        func number() throws -> UInt64 {
                            guard valueCursor + 8 <= extraCursor + length else { throw NativeTorchCheckpoint.failure("truncated ZIP64 size") }
                            defer { valueCursor += 8 }; return bytes.u64(valueCursor)
                        }
                        if expanded == UInt32.max { expanded = try number() }
                        if compressed == UInt32.max { compressed = try number() }
                        if offset == UInt32.max { offset = try number() }
                        found = true
                    }
                    extraCursor += length
                }
                guard found else { throw NativeTorchCheckpoint.failure("missing ZIP64 sizes") }
            }
            guard let compressedSize = Int(exactly: compressed), let expandedSize = Int(exactly: expanded), let memberOffset = Int(exactly: offset),
                  expandedSize <= 8_000_000_000, memberOffset <= start - 30, compressedSize <= start - memberOffset - 30 else { throw NativeTorchCheckpoint.failure("ZIP member exceeds file bounds") }
            entries[name] = Entry(name: name, method: method, crc: bytes.u32(cursor + 16), compressed: compressedSize, size: expandedSize, offset: memberOffset)
            cursor += recordSize
        }
        guard cursor == start + size else { throw NativeTorchCheckpoint.failure("ZIP directory size differs from entries") }
        self.entries = entries
    }
    func read(_ name: String) throws -> Data {
        try Task.checkCancellation()
        guard let entry = entries[name] else { throw NativeTorchCheckpoint.failure("missing storage \(name)") }
        let cursor = entry.offset
        guard bytes.u32(cursor) == 0x04034b50, bytes.u16(cursor + 8) == entry.method, bytes.u16(cursor + 6) & 1 == 0 else { throw NativeTorchCheckpoint.failure("invalid ZIP local member") }
        let nameSize = Int(bytes.u16(cursor + 26)), extraSize = Int(bytes.u16(cursor + 28))
        let start = cursor + 30 + nameSize + extraSize
        guard start <= bytes.count, entry.compressed <= bytes.count - start,
              String(data: bytes.subdata(in: cursor + 30..<cursor + 30 + nameSize), encoding: .utf8) == name else { throw NativeTorchCheckpoint.failure("ZIP local name or size differs") }
        let compressed = bytes.subdata(in: start..<start + entry.compressed)
        let output: Data
        if entry.method == 0 {
            guard entry.compressed == entry.size else { throw NativeTorchCheckpoint.failure("stored ZIP member size differs") }; output = compressed
        } else {
            guard entry.compressed <= Int(UInt32.max), entry.size <= Int(UInt32.max) else { throw NativeTorchCheckpoint.failure("deflated member exceeds native decoder limits") }
            let capacity = max(1, entry.size)
            var inflated = Data(count: capacity), stream = z_stream()
            guard inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else { throw NativeTorchCheckpoint.failure("ZIP decoder initialization failed") }
            defer { inflateEnd(&stream) }
            let status = inflated.withUnsafeMutableBytes { destination in compressed.withUnsafeBytes { source in
                stream.next_in = UnsafeMutablePointer(mutating: source.bindMemory(to: Bytef.self).baseAddress)
                stream.avail_in = uInt(compressed.count)
                stream.next_out = destination.bindMemory(to: Bytef.self).baseAddress
                stream.avail_out = uInt(capacity)
                return inflate(&stream, Z_FINISH)
            } }
            guard status == Z_STREAM_END, stream.total_out == entry.size, stream.total_in == entry.compressed else { throw NativeTorchCheckpoint.failure("ZIP deflate size differs") }
            inflated.count = entry.size; output = inflated
        }
        let crc = output.withUnsafeBytes { crc32_z(0, $0.bindMemory(to: Bytef.self).baseAddress, output.count) }
        guard UInt32(crc) == entry.crc else { throw NativeTorchCheckpoint.failure("ZIP storage checksum changed") }
        return output
    }
}

private struct TorchStorage { let key: String; let dtype: String; let count: Int }
private struct TorchTensor { let storage: TorchStorage; let offset: Int; let shape: [Int]; let stride: [Int]; let dtype: String? }
private final class PickleValue {
    enum Kind {
        case mark, none, boolean(Bool), integer(Int), number(Double), string(String), bytes(Data)
        case tuple([PickleValue]), list([PickleValue]), dictionary([(PickleValue, PickleValue)])
        case global(String), storage(TorchStorage), tensor(TorchTensor)
    }
    var kind: Kind
    init(_ kind: Kind) { self.kind = kind }
}

private final class TorchPickle {
    let bytes: Data
    var cursor = 0, stack: [PickleValue] = [], memo: [Int: PickleValue] = [:], instructions = 0
    init(_ bytes: Data) { self.bytes = bytes }
    func decode() throws -> PickleValue {
        while cursor < bytes.count {
            instructions += 1
            guard instructions <= 5_000_000, stack.count <= 1_000_000, memo.count <= 1_000_000 else { throw bad("metadata complexity exceeds limits") }
            if instructions & 0x3fff == 0 { try Task.checkCancellation() }
            let opcode = try byte()
            switch opcode {
            case 0x80: guard (2...5).contains(try byte()) else { throw bad("unsupported pickle protocol") }
            case 0x95: let size = try uint64(); guard size <= UInt64(bytes.count - cursor) else { throw bad("truncated pickle frame") }
            case 0x2e:
                guard stack.count == 1, cursor == bytes.count else { throw bad("pickle STOP has extra objects or bytes") }; return stack[0]
            case 0x28: stack.append(PickleValue(.mark))
            case 0x4e: stack.append(PickleValue(.none))
            case 0x88: stack.append(PickleValue(.boolean(true)))
            case 0x89: stack.append(PickleValue(.boolean(false)))
            case 0x4b: stack.append(PickleValue(.integer(Int(try byte()))))
            case 0x4d: stack.append(PickleValue(.integer(Int(try uint16()))))
            case 0x4a: stack.append(PickleValue(.integer(Int(Int32(bitPattern: try uint32())))))
            case 0x8a, 0x8b:
                let size = opcode == 0x8a ? Int(try byte()) : try length32()
                let data = try read(size); guard size <= 8 else { throw bad("integer exceeds native precision") }
                var value: UInt64 = 0
                for (offset, code) in data.enumerated() { value |= UInt64(code) << (offset * 8) }
                if let last = data.last, last & 128 != 0, size < 8 { value |= UInt64.max << (size * 8) }
                guard let integer = Int(exactly: Int64(bitPattern: value)) else { throw bad("integer overflow") }
                stack.append(PickleValue(.integer(integer)))
            case 0x47:
                let data = try read(8); let bits = data.reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
                stack.append(PickleValue(.number(Double(bitPattern: bits))))
            case 0x58: stack.append(PickleValue(.string(try string(length32()))))
            case 0x8c: stack.append(PickleValue(.string(try string(Int(byte())))))
            case 0x8d: guard let size = Int(exactly: try uint64()) else { throw bad("string length overflows") }; stack.append(PickleValue(.string(try string(size))))
            case 0x42: stack.append(PickleValue(.bytes(try read(length32()))))
            case 0x43: stack.append(PickleValue(.bytes(try read(Int(byte())))))
            case 0x29: stack.append(PickleValue(.tuple([])))
            case 0x5d: stack.append(PickleValue(.list([])))
            case 0x7d: stack.append(PickleValue(.dictionary([])))
            case 0x74: stack.append(PickleValue(.tuple(try marked())))
            case 0x6c: stack.append(PickleValue(.list(try marked())))
            case 0x64: stack.append(PickleValue(.dictionary(try pairs(marked()))))
            case 0x85, 0x86, 0x87:
                let count = Int(opcode - 0x84); guard stack.count >= count else { throw bad("tuple stack underflow") }
                let values = Array(stack.suffix(count)); stack.removeLast(count); stack.append(PickleValue(.tuple(values)))
            case 0x61: let value = try pop(); guard let top = stack.last, case .list(var values) = top.kind else { throw bad("APPEND target is not a list") }; values.append(value); top.kind = .list(values)
            case 0x65: let values = try marked(); guard let top = stack.last, case .list(var old) = top.kind else { throw bad("APPENDS target is not a list") }; old += values; top.kind = .list(old)
            case 0x73: let value = try pop(), key = try pop(); try set([(key, value)])
            case 0x75: try set(pairs(marked()))
            case 0x71: try remember(Int(byte()))
            case 0x72: try remember(length32())
            case 0x94: try remember(memo.count)
            case 0x68: stack.append(try recalled(Int(byte())))
            case 0x6a: stack.append(try recalled(length32()))
            case 0x63: let module = try line(), name = try line(); stack.append(try global(module + "." + name))
            case 0x93:
                let name = try text(pop()), module = try text(pop()); stack.append(try global(module + "." + name))
            case 0x51: stack.append(try storage(pop()))
            case 0x52:
                let arguments = try tuple(pop()), callable = try pop(); stack.append(try reduce(callable, arguments))
            case 0x62:
                let state = try pop(); guard let instance = stack.last, case .dictionary = instance.kind, case .dictionary(let attributes) = state.kind,
                      attributes.allSatisfy({ pair in if case .string(let key) = pair.0.kind { return key == "_metadata" }; return false }) else { throw bad("object BUILD is not permitted") }
                try metadataOnly(state, depth: 0)
            default: throw bad(String(format: "unsupported executable or data pickle opcode 0x%02x", opcode))
            }
        }
        throw bad("pickle is incomplete")
    }
    private func global(_ name: String) throws -> PickleValue {
        guard Self.storageTypes[name] != nil || Self.dtypes[name] != nil || ["collections.OrderedDict", "torch._utils._rebuild_tensor", "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor_v3"].contains(name) else { throw bad("global \(name) is not a data-only tensor operation") }
        return PickleValue(.global(name))
    }
    private func storage(_ value: PickleValue) throws -> PickleValue {
        let values = try tuple(value)
        guard values.count == 5, try text(values[0]) == "storage", case .global(let type) = values[1].kind,
              let dtype = Self.storageTypes[type] else { throw bad("persistent ID is not an approved tensor storage") }
        let key = try text(values[2]), location = try text(values[3]), count = try integer(values[4])
        guard !key.isEmpty, key.allSatisfy({ $0.isASCII && $0.isNumber }), count >= 0,
              location == "cpu" || location.hasPrefix("cuda:") || location == "mps" else { throw bad("invalid tensor storage identity") }
        return PickleValue(.storage(TorchStorage(key: key, dtype: dtype, count: count)))
    }
    private func reduce(_ callable: PickleValue, _ values: [PickleValue]) throws -> PickleValue {
        guard case .global(let name) = callable.kind else { throw bad("REDUCE callable is not approved") }
        if name == "collections.OrderedDict" {
            if values.isEmpty { return PickleValue(.dictionary([])) }
            guard values.count == 1, case .list(let items) = values[0].kind else { throw bad("invalid OrderedDict data") }
            return PickleValue(.dictionary(try items.map { item in let pair = try tuple(item); guard pair.count == 2 else { throw bad("invalid OrderedDict pair") }; return (pair[0], pair[1]) }))
        }
        guard name.hasPrefix("torch._utils._rebuild_tensor"), values.count == (name.hasSuffix("_v3") ? 7 : name.hasSuffix("_v2") ? 6 : 4),
              case .storage(let storage) = values[0].kind else { throw bad("invalid tensor rebuild data") }
        let offset = try integer(values[1]), shape = try tuple(values[2]).map(integer), stride = try tuple(values[3]).map(integer)
        if values.count >= 6 {
            guard case .boolean = values[4].kind, case .dictionary(let hooks) = values[5].kind, hooks.isEmpty else { throw bad("tensor hooks are not permitted") }
        }
        var dtype: String?
        if values.count == 7 {
            guard case .global(let type) = values[6].kind, let selected = Self.dtypes[type] else { throw bad("unknown tensor dtype") }; dtype = selected
        }
        guard storage.dtype != "UNTYPED" || dtype != nil else { throw bad("untyped storage must declare dtype") }
        return PickleValue(.tensor(TorchTensor(storage: storage, offset: offset, shape: shape, stride: stride, dtype: dtype)))
    }
    private func metadataOnly(_ node: PickleValue, depth: Int) throws {
        guard depth <= 64 else { throw bad("module metadata is too deeply nested") }
        switch node.kind {
        case .none, .boolean, .integer, .number, .string: break
        case .tuple(let items), .list(let items): for item in items { try metadataOnly(item, depth: depth + 1) }
        case .dictionary(let pairs): for (key, value) in pairs { _ = try text(key); try metadataOnly(value, depth: depth + 1) }
        default: throw bad("module metadata contains non-data objects")
        }
    }
    private func set(_ pairs: [(PickleValue, PickleValue)]) throws {
        guard let target = stack.last, case .dictionary(var old) = target.kind else { throw bad("SETITEM target is not a dictionary") }
        for pair in pairs {
            let key = try text(pair.0)
            guard !old.contains(where: { if case .string(let existing) = $0.0.kind { return existing == key }; return false }) else { throw bad("duplicate state dictionary key") }
            old.append(pair)
        }
        target.kind = .dictionary(old)
    }
    private func pairs(_ values: [PickleValue]) throws -> [(PickleValue, PickleValue)] {
        guard values.count % 2 == 0 else { throw bad("unpaired dictionary data") }
        return stride(from: 0, to: values.count, by: 2).map { (values[$0], values[$0 + 1]) }
    }
    private func marked() throws -> [PickleValue] {
        guard let mark = stack.lastIndex(where: { if case .mark = $0.kind { return true }; return false }) else { throw bad("missing pickle mark") }
        let values = Array(stack[(mark + 1)...]); stack.removeSubrange(mark...); return values
    }
    private func remember(_ index: Int) throws {
        guard index >= 0, index <= 1_000_000, let value = stack.last, memo[index] == nil else { throw bad("invalid pickle memo") }; memo[index] = value
    }
    private func recalled(_ index: Int) throws -> PickleValue { guard let value = memo[index] else { throw bad("missing pickle memo") }; return value }
    private func pop() throws -> PickleValue { guard let value = stack.popLast() else { throw bad("pickle stack underflow") }; return value }
    private func tuple(_ value: PickleValue) throws -> [PickleValue] { guard case .tuple(let values) = value.kind else { throw bad("expected tuple data") }; return values }
    private func text(_ value: PickleValue) throws -> String { guard case .string(let value) = value.kind else { throw bad("expected string data") }; return value }
    private func integer(_ value: PickleValue) throws -> Int { guard case .integer(let value) = value.kind else { throw bad("expected integer tensor metadata") }; return value }
    private func byte() throws -> UInt8 { guard cursor < bytes.count else { throw bad("truncated pickle") }; defer { cursor += 1 }; return bytes[cursor] }
    private func read(_ count: Int) throws -> Data { guard count >= 0, count <= bytes.count - cursor else { throw bad("truncated pickle data") }; defer { cursor += count }; return bytes.subdata(in: cursor..<cursor + count) }
    private func string(_ count: Int) throws -> String { guard let value = String(data: try read(count), encoding: .utf8) else { throw bad("invalid UTF-8 metadata") }; return value }
    private func line() throws -> String { guard let end = bytes[cursor...].firstIndex(of: 10) else { throw bad("truncated pickle global") }; let value = try string(end - cursor); cursor += 1; return value }
    private func uint16() throws -> UInt16 { try read(2).u16(0) }
    private func uint32() throws -> UInt32 { try read(4).u32(0) }
    private func uint64() throws -> UInt64 { try read(8).u64(0) }
    private func length32() throws -> Int { Int(try uint32()) }
    private func bad(_ message: String) -> NativeCheckpointError { NativeTorchCheckpoint.failure(message) }
    private static let storageTypes = ["torch.DoubleStorage": "F64", "torch.FloatStorage": "F32", "torch.HalfStorage": "F16", "torch.BFloat16Storage": "BF16",
        "torch.LongStorage": "I64", "torch.IntStorage": "I32", "torch.ShortStorage": "I16", "torch.CharStorage": "I8", "torch.ByteStorage": "U8", "torch.BoolStorage": "BOOL", "torch.storage.UntypedStorage": "UNTYPED"]
    private static let dtypes = ["torch.float64": "F64", "torch.float32": "F32", "torch.float16": "F16", "torch.bfloat16": "BF16", "torch.int64": "I64", "torch.int32": "I32", "torch.int16": "I16", "torch.int8": "I8", "torch.uint8": "U8", "torch.bool": "BOOL"]
}

private extension Data {
    func u16(_ offset: Int) -> UInt16 { withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self)) } }
    func u32(_ offset: Int) -> UInt32 { withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) } }
    func u64(_ offset: Int) -> UInt64 { withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)) } }
}
