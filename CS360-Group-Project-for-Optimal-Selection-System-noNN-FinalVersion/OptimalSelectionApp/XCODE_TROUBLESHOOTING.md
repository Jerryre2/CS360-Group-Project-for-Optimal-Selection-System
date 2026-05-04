# Xcode Troubleshooting

## Why Xcode Keeps Showing "Preparing Editor Functionality"

This message usually means Xcode is still preparing editor services such as:

- Swift Package manifest parsing.
- SourceKit indexing.
- Code completion database.
- DerivedData cache.
- Source control integration.

It does not necessarily mean the app cannot build.

For this project, the command-line build already succeeds:

```bash
cd /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN/OptimalSelectionApp
swift build
```

If this prints `Build complete!`, the Swift code itself is valid.

## Most Likely Cause

Xcode may have opened the entire noNN repository instead of the Swift Package file:

```text
CS360-Group-Project-for-Optimal-Selection-System-noNN
```

When that happens, Xcode may index Python files such as:

```text
optimal_samples_system/validation.py
```

The Swift App target may not be selected correctly, so Run appears unavailable or stuck.

## Correct Way To Open

Close the current Xcode window first.

Then run:

```bash
cd /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN/OptimalSelectionApp
./open_in_xcode.sh
```

Or manually:

```bash
open -a Xcode /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN/OptimalSelectionApp/OptimalSelectionApp.xcodeproj
```

In Xcode:

1. Select scheme `OptimalSelectionApp`.
2. Select destination `My Mac`.
3. Wait until indexing finishes.
4. Press `Cmd + R`.

## If It Still Keeps Spinning

Try clearing only this Swift package's local build metadata:

```bash
cd /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN/OptimalSelectionApp
rm -rf .build .swiftpm
swift build
./open_in_xcode.sh
```

If Xcode itself is stuck, clear DerivedData:

```bash
rm -rf ~/Library/Developer/Xcode/DerivedData/OptimalSelectionApp-*
```

Then reopen:

```bash
./open_in_xcode.sh
```

## Source Control Warning Is Not Related

The message:

```text
No Author / Git Author Settings
```

only affects Git commits inside Xcode. It does not prevent building or running the app.

## Emergency Run Without Xcode

If Xcode UI keeps indexing, you can still build from Terminal:

```bash
cd /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN/OptimalSelectionApp
swift run
```

This should launch the SwiftUI app directly if macOS allows the package executable to start as a GUI process.

## Better Fix Applied

To avoid Xcode failing to load the Swift Package folder, this project now also includes a standard Xcode project:

```text
OptimalSelectionApp.xcodeproj
```

This project has already been validated with:

```bash
xcodebuild -project OptimalSelectionApp.xcodeproj -scheme OptimalSelectionApp -configuration Debug -destination 'platform=macOS' build
```

Observed result:

```text
** BUILD SUCCEEDED **
```
