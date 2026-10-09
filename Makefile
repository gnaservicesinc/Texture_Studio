PYTHON_BASE ?= /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14
PYTHON ?= $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/python,$(PYTHON_BASE))
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

.PHONY: check-toolchain setup configure build package install install-if-closed release-check run gui studio debug test test-native test-python smoke clean

check-toolchain:
	./script/check_toolchain.sh

# Extraction and material-model development environment.
setup:
	$(PYTHON_BASE) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -c requirements-release-macos.txt \
		-r requirements.txt -r requirements-depth.txt 'datasets>=4,<6'
	.venv/bin/python -m pip check
	.venv/bin/python -m pip install --no-deps -e .

configure: check-toolchain
	xcodebuild -list -project "$(XCODE_PROJECT)"

build: check-toolchain
	$(XCODEBUILD) build
	./script/stage_material_apps.sh "$(APP_BUNDLE)"

package: build
	./scripts/package_macos.sh "$(APP_BUNDLE)" "$(CURDIR)/dist"

install: package
	"$(PYTHON)" scripts/install_macos.py "$(APP_BUNDLE)" --destdir "$(DESTDIR)"

install-if-closed: package
	"$(PYTHON)" scripts/install_macos.py "$(APP_BUNDLE)" --destdir "$(DESTDIR)" --if-closed

release-check: package
	"$(PYTHON)" scripts/check_release_version.py

run gui studio:
	./script/build_and_run.sh

debug:
	./script/build_and_run.sh --debug

test: test-native test-python

test-native: check-toolchain
	xcodebuild -project "$(XCODE_PROJECT)" -scheme "$(XCODE_SCHEME)" \
		-configuration Debug -destination 'platform=macOS,arch=arm64' \
		-derivedDataPath "$(DERIVED_DATA)" test

test-python:
	PYTHONPATH=$(CURDIR)/src "$(PYTHON)" -m unittest discover -s verification -v

smoke: build
	"$(APP_BUNDLE)/Contents/MacOS/Texture Studio" --smoke-test

clean: check-toolchain
	$(XCODEBUILD) clean
