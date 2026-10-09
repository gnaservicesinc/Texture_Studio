import Foundation

enum WorkbenchResult {
    static func decode<T: Decodable>(_ type: T.Type, output: String) throws -> T {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        if let result = try? decoder.decode(type, from: Data(output.utf8)) { return result }
        for line in output.split(separator: "\n").reversed() {
            if let decoded = try? decoder.decode(type, from: Data(line.utf8)) { return decoded }
        }
        throw StudioError("The native material operation did not return a valid result. See the operation log.")
    }
}
