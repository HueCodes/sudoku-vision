import SwiftUI

struct ContentView: View {
    @State private var showManualEntry = false
    @State private var showPipelineTest = false

    var body: some View {
        NavigationStack {
            CameraScanView()
                .toolbar {
                    #if DEBUG
                    ToolbarItem(placement: .topBarLeading) {
                        Button {
                            showPipelineTest = true
                        } label: {
                            Image(systemName: "flask")
                        }
                    }
                    #endif

                    ToolbarItem(placement: .topBarTrailing) {
                        Button {
                            showManualEntry = true
                        } label: {
                            Image(systemName: "keyboard")
                        }
                    }
                }
                .sheet(isPresented: $showManualEntry) {
                    ManualEntryView()
                }
                .sheet(isPresented: $showPipelineTest) {
                    PipelineTestView()
                }
        }
    }
}

#Preview {
    ContentView()
}
