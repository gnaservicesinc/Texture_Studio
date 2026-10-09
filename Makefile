CONFIGURATION ?= Release
DERIVED_DATA ?= $(CURDIR)/build/TextureStudio
DESTDIR ?= /
XCODE_PROJECT := native/TextureStudio/TextureStudio.xcodeproj
XCODE_SCHEME := TextureStudio
APP_BUNDLE := $(DERIVED_DATA)/Build/Products/$(CONFIGURATION)/Texture Studio.app
XCODEBUILD = xcodebuild -project "$(XCODE_PROJECT)" -scheme "$(XCODE_SCHEME)" \
	-configuration "$(CONFIGURATION)" -destination 'platform=macOS,arch=arm64' \
	-derivedDataPath "$(DERIVED_DATA)"
.DEFAULT_GOAL := build

.PHONY: check-toolchain configure build package install install-if-closed release-check run gui studio debug test test-native test-build-tools smoke clean

check-toolchain:
	./script/check_toolchain.sh

configure: check-toolchain
	xcodebuild -list -project "$(XCODE_PROJECT)"

build: check-toolchain
	$(XCODEBUILD) build
	./script/stage_material_apps.sh "$(APP_BUNDLE)"

package: build
	./scripts/package_macos.sh "$(APP_BUNDLE)" "$(CURDIR)/dist"

install: package
	/usr/bin/xcrun swift script/install_macos.swift "$(APP_BUNDLE)" --destdir "$(DESTDIR)"

install-if-closed: package
	/usr/bin/xcrun swift script/install_macos.swift "$(APP_BUNDLE)" --destdir "$(DESTDIR)" --if-closed

release-check: package
	/usr/bin/xcrun swift script/check_release_version.swift

run gui studio:
	./script/build_and_run.sh

debug:
	./script/build_and_run.sh --debug

test: test-native

test-native: check-toolchain test-build-tools
	xcodebuild -project "$(XCODE_PROJECT)" -scheme "$(XCODE_SCHEME)" \
		-configuration Debug -destination 'platform=macOS,arch=arm64' \
		-derivedDataPath "$(DERIVED_DATA)" test

test-build-tools: check-toolchain
	./script/test_build_tools.sh

smoke: build
	"$(APP_BUNDLE)/Contents/MacOS/Texture Studio" --smoke-test

clean: check-toolchain
	$(XCODEBUILD) clean
