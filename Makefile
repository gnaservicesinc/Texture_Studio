PYTHON_BASE ?= /Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14
PYTHON ?= $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/python,$(PYTHON_BASE))
QT_CMAKE ?= $(shell ls -d /opt/Qt/6.*/macos/lib/cmake/Qt6 2>/dev/null | sort -V | tail -1)
BUILD_DIR ?= build
BUILD_TYPE ?= Release
DESTDIR ?= /
CMAKE_ARGS ?=

.PHONY: setup configure build package install gui studio test smoke clean

setup:
	$(PYTHON_BASE) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -c requirements-release-macos.txt \
		-r requirements.txt -r requirements-depth.txt 'datasets>=4,<6'
	.venv/bin/python -m pip check
	.venv/bin/python -m pip install --no-deps -e .

configure:
	cmake -S . -B $(BUILD_DIR) -G Ninja \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_PREFIX_PATH="$(QT_CMAKE)" \
		-DIPDE_PYTHON_EXECUTABLE="$(PYTHON)" $(CMAKE_ARGS)

build: configure
	cmake --build $(BUILD_DIR)

package: build
	cmake --build $(BUILD_DIR) --target distribution

install: package
	"$(PYTHON)" scripts/install_macos.py "$(BUILD_DIR)/dist/IPDE Studio.app" --destdir "$(DESTDIR)"

gui: build
	open "$(CURDIR)/$(BUILD_DIR)/IPDE.app"

studio: build
	open "$(CURDIR)/$(BUILD_DIR)/IPDE Studio.app"

test:
	PYTHONPATH=$(CURDIR)/src $(PYTHON) -m unittest discover -s verification -v

smoke: build
	"$(CURDIR)/$(BUILD_DIR)/IPDE.app/Contents/MacOS/IPDE" --smoke-test

clean:
	cmake -E remove_directory $(BUILD_DIR)
