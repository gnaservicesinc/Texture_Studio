import SwiftUI

struct PreparationSettingsView: View {
    @Bindable var store: WorkbenchStore

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Stepper("Preparation workers: \(store.preparationWorkers)", value: $store.preparationWorkers,
                    in: 1...store.resources.availableProcessorCount)
                .accessibilityIdentifier("preparationWorkers")
                .disabled(store.isBusy)
            Text("Defaults to all \(store.resources.availableProcessorCount) available CPU cores. Preparation uses fewer workers when the maps need more memory.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }
}
