import XCTest
@testable import TextureStudio

final class NumericFieldTests: XCTestCase {
    func testTypingQuickCheckIntervalAcceptsWholeValueAndBoundaries() {
        XCTAssertEqual(NumericTextEditing.value(from: "10000", in: 1...10_000), 10_000)
        XCTAssertEqual(NumericTextEditing.value(from: "1", in: 1...10_000), 1)
        XCTAssertNil(NumericTextEditing.value(from: "0", in: 1...10_000))
        XCTAssertNil(NumericTextEditing.value(from: "10001", in: 1...10_000))
        XCTAssertNil(NumericTextEditing.value(from: "1.5", in: 1...10_000))
        XCTAssertEqual(NumericTextEditing.value(from: "0", in: 0...100_000), 0,
                       "Zero retains the on-request checkpoint setting.")
    }

    func testSmallPhysicalValuesAndScientificNotationKeepPrecision() throws {
        let relief: Double = try XCTUnwrap(NumericTextEditing.value(from: "0.005", atLeast: 0))
        XCTAssertEqual(relief, 0.005)
        let microscopicRelief: Double = try XCTUnwrap(NumericTextEditing.value(from: "1e-9", atLeast: 0))
        XCTAssertEqual(microscopicRelief, 0.000000001)
        let expected = Double(bitPattern: 0x3fd5555555555555)
        let roundTrip: Double = try XCTUnwrap(NumericTextEditing.value(from: String(expected)))
        XCTAssertEqual(roundTrip.bitPattern, expected.bitPattern,
                       "Displaying and editing numeric settings must not round to two decimal places.")
        let lighting: Float = try XCTUnwrap(NumericTextEditing.value(from: "0.00125", in: Float(0)...Float(1)))
        XCTAssertEqual(lighting, 0.00125)
    }

    func testIncompleteAndNonfiniteDraftsDoNotBecomeSettings() {
        for text in ["", "-", ".", "1e", "nan", "inf", "-inf", "1e999", "abc"] {
            let value: Double? = NumericTextEditing.value(from: text)
            XCTAssertNil(value, "Invalid draft \(text) must retain the previous valid setting.")
        }
        XCTAssertNil(NumericTextEditing.value(from: "-0.005", atLeast: Double(0)))
        XCTAssertNil(NumericTextEditing.value(from: "0", greaterThan: Double(0)))
        XCTAssertEqual(NumericTextEditing.value(from: "0", atLeast: Double(0)), 0)
    }

    func testDecimalCommaAndWhitespaceCanBeEnteredDirectly() {
        let value: Double? = NumericTextEditing.value(from: " 0,005 \n", atLeast: 0, decimalSeparator: ",")
        XCTAssertEqual(value, 0.005)
        let scientific: Double? = NumericTextEditing.value(from: "1,25e-3", decimalSeparator: ",")
        XCTAssertEqual(scientific, 0.00125)
    }

    func testAdapterWeightsRetainSignedUnboundedValues() {
        let negative: Double? = NumericTextEditing.value(from: "-0.75")
        let zero: Double? = NumericTextEditing.value(from: "0")
        let additive: Double? = NumericTextEditing.value(from: "2.5")
        XCTAssertEqual(negative, -0.75)
        XCTAssertEqual(zero, 0)
        XCTAssertEqual(additive, 2.5)
    }

    func testSliderChangeWhileTextFieldIsFocusedReplacesOldDraftBeforeSubmit() throws {
        var draft = NumericTextDraft(value: Double(0.2))
        draft.text = "0.2000"
        draft.accept(0.2)
        draft.synchronize(with: 0.2, isFocused: true)
        XCTAssertEqual(draft.text, "0.2000", "Typing echoes must preserve the user's unfinished spelling.")

        draft.synchronize(with: 0.8, isFocused: true)
        let submitted: Double = try XCTUnwrap(NumericTextEditing.value(from: draft.text, in: 0...1))
        XCTAssertEqual(submitted, 0.8, "Submitting must retain the newer slider value, not restore the old text.")
        XCTAssertEqual(draft.lastAcceptedValue, 0.8)
    }

    func testExternalResetReplacesIncompleteFocusedDraftAndOptionalFocalLength() {
        var draft = NumericTextDraft(value: Double(1200))
        draft.text = "1e"
        draft.synchronize(with: 800, isFocused: true)
        XCTAssertEqual(draft.text, "800.0")
        draft.text = "-"
        draft.synchronize(with: nil, isFocused: true)
        XCTAssertEqual(draft.text, "", "Resetting camera focal length must restore Automatic even with text focus.")
        XCTAssertNil(draft.lastAcceptedValue)
    }
}
