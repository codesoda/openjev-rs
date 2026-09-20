#!/bin/sh
# Install a verified prebuilt openjev release. This file deliberately contains
# definitions only until the final main invocation so a truncated curl | sh
# download cannot start an installation.

set -eu

REPO="codesoda/openjev-rs"
REPO_URL="https://github.com/${REPO}"
MEMBER_FILES="openjev LICENSE THIRD_PARTY.md THIRD_PARTY_LICENSES.html RUST-COPYRIGHT-library.html colored-3.1.1.crate option-ext-0.2.0.crate README.md SERVE.md BUILD-INFO.json"
TEMP_DIR=""
INSTALL_STAGE=""
LINK_STAGE=""
LOCK_DIR=""
LOCK_HELD=0

say() {
    printf '%s\n' "$*" >&2
}

fail() {
    say "openjev installer: error: $*"
    exit 1
}

usage() {
    cat >&2 <<'EOF'
Usage: install.sh [--version vMAJOR.MINOR.PATCH]

Install the latest openjev release, or the exact release selected by --version.
Supported platforms: macOS 14+ on Apple silicon, and Linux x86_64 with glibc 2.35+.
EOF
}

cleanup() {
    if [ -n "$LINK_STAGE" ] && [ -d "$LINK_STAGE" ]; then
        rm -rf "$LINK_STAGE"
    fi
    if [ -n "$INSTALL_STAGE" ] && [ -d "$INSTALL_STAGE" ]; then
        rm -rf "$INSTALL_STAGE"
    fi
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
    if [ "$LOCK_HELD" -eq 1 ] && [ -n "$LOCK_DIR" ]; then
        rmdir "$LOCK_DIR" 2>/dev/null || :
        LOCK_HELD=0
    fi
}

on_signal() {
    trap - 0 1 2 15
    cleanup
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command '$1' was not found"
}

is_valid_tag() {
    printf '%s\n' "$1" | LC_ALL=C grep -Eq '^v[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$'
}

is_numeric_version() {
    printf '%s\n' "$1" | LC_ALL=C grep -Eq '^[0-9]+\.[0-9]+(\.[0-9]+)?$'
}

version_at_least() {
    awk -v actual="$1" -v minimum="$2" 'BEGIN {
        split(actual, a, "."); split(minimum, m, ".")
        for (i = 1; i <= 3; i++) {
            av = a[i] + 0; mv = m[i] + 0
            if (av > mv) exit 0
            if (av < mv) exit 1
        }
        exit 0
    }'
}

check_home() {
    case "${HOME:-}" in
        /*) ;;
        *) fail "HOME must name an existing absolute directory" ;;
    esac
    [ -d "$HOME" ] || fail "HOME must name an existing absolute directory"
    [ ! -L "$HOME" ] || fail "refusing symlinked HOME: $HOME"
    PHYSICAL_HOME=$(CDPATH=; cd "$HOME" 2>/dev/null && pwd -P) || fail "cannot access HOME: $HOME"
    [ "$PHYSICAL_HOME" = "$HOME" ] || fail "HOME contains a symlinked path component: $HOME"
}

ensure_real_directory() {
    directory=$1
    if [ -L "$directory" ]; then
        fail "refusing symlinked installation directory: $directory"
    fi
    if [ -e "$directory" ]; then
        [ -d "$directory" ] || fail "installation path is not a directory: $directory"
    else
        mkdir "$directory" || fail "could not create installation directory: $directory"
    fi
    [ ! -L "$directory" ] && [ -d "$directory" ] || fail "unsafe installation directory: $directory"
}

prepare_directories() {
    OPENJEV_HOME="$HOME/.openjev"
    BIN_DIR="$OPENJEV_HOME/bin"
    LOCAL_HOME="$HOME/.local"
    LINK_DIR="$LOCAL_HOME/bin"
    ACTIVE_LINK="$BIN_DIR/openjev"
    COMMAND_LINK="$LINK_DIR/openjev"

    ensure_real_directory "$OPENJEV_HOME"
    LOCK_DIR="$OPENJEV_HOME/install.lock"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        fail "another openjev installer is running (lock: $LOCK_DIR)"
    fi
    LOCK_HELD=1
    ensure_real_directory "$BIN_DIR"
    ensure_real_directory "$LOCAL_HOME"
    ensure_real_directory "$LINK_DIR"
}

has_canonical_parent() {
    candidate_parent=${1%/*}
    [ -d "$candidate_parent" ] && [ ! -L "$candidate_parent" ] || return 1
    physical_parent=$(CDPATH=; cd "$candidate_parent" 2>/dev/null && pwd -P) || return 1
    [ "$physical_parent" = "$candidate_parent" ]
}

valid_new_active_link() {
    link_target=$1
    prefix="$BIN_DIR/openjev-v"
    case "$link_target" in
        "$prefix"*-aarch64-apple-darwin/openjev)
            value=${link_target#"$prefix"}
            value=${value%-aarch64-apple-darwin/openjev}
            ;;
        "$prefix"*-x86_64-unknown-linux-gnu/openjev)
            value=${link_target#"$prefix"}
            value=${value%-x86_64-unknown-linux-gnu/openjev}
            ;;
        *) return 1 ;;
    esac
    is_valid_tag "v$value" && has_canonical_parent "$link_target"
}

valid_previous_link() {
    link_target=$1
    prefix="$HOME/.local/share/openjev/releases/"
    case "$link_target" in
        "$prefix"*) ;;
        *) return 1 ;;
    esac
    rest=${link_target#"$prefix"}
    old_tag=${rest%%/*}
    [ "$old_tag" != "$rest" ] || return 1
    is_valid_tag "$old_tag" || return 1
    rest=${rest#*/}
    for old_target in aarch64-apple-darwin x86_64-unknown-linux-gnu; do
        if [ "$rest" = "openjev-${old_tag}-${old_target}/openjev" ]; then
            [ -f "$link_target" ] && [ ! -L "$link_target" ] && has_canonical_parent "$link_target"
            return
        fi
    done
    return 1
}

check_existing_links() {
    if [ -L "$ACTIVE_LINK" ]; then
        active_target=$(readlink "$ACTIVE_LINK") || fail "could not inspect $ACTIVE_LINK"
        valid_new_active_link "$active_target" || fail "refusing unrelated active link: $ACTIVE_LINK -> $active_target"
        [ -f "$active_target" ] && [ ! -L "$active_target" ] || fail "refusing broken or non-regular active link: $ACTIVE_LINK"
    elif [ -e "$ACTIVE_LINK" ]; then
        fail "refusing to replace unrelated path: $ACTIVE_LINK"
    fi

    REPLACE_COMMAND_LINK=1
    if [ -L "$COMMAND_LINK" ]; then
        command_target=$(readlink "$COMMAND_LINK") || fail "could not inspect $COMMAND_LINK"
        if [ "$command_target" = "$ACTIVE_LINK" ]; then
            REPLACE_COMMAND_LINK=0
        elif valid_previous_link "$command_target"; then
            REPLACE_COMMAND_LINK=1
        else
            fail "refusing unrelated command link: $COMMAND_LINK -> $command_target"
        fi
    elif [ -e "$COMMAND_LINK" ]; then
        fail "refusing to replace unrelated path: $COMMAND_LINK"
    fi
}

detect_target() {
    os=$(uname -s) || fail "could not detect operating system"
    arch=$(uname -m) || fail "could not detect architecture"
    case "$os:$arch" in
        Darwin:arm64|Darwin:aarch64)
            require_command sw_vers
            mac_version=$(sw_vers -productVersion) || fail "could not determine macOS version"
            is_numeric_version "$mac_version" || fail "could not determine macOS version"
            version_at_least "$mac_version" 14.0.0 || fail "macOS 14 or newer is required (found $mac_version)"
            TARGET="aarch64-apple-darwin"
            require_command xattr
            ;;
        Linux:x86_64|Linux:amd64)
            require_command getconf
            libc=$(getconf GNU_LIBC_VERSION 2>/dev/null) || fail "Linux releases require glibc 2.35 or newer"
            case "$libc" in
                "glibc "*) libc_version=${libc#"glibc "} ;;
                *) fail "Linux releases require glibc 2.35 or newer" ;;
            esac
            is_numeric_version "$libc_version" || fail "could not determine glibc version"
            version_at_least "$libc_version" 2.35.0 || fail "glibc 2.35 or newer is required (found $libc_version)"
            TARGET="x86_64-unknown-linux-gnu"
            ;;
        *) fail "unsupported platform $os/$arch; supported platforms are macOS 14+ arm64 and Linux x86_64 with glibc 2.35+" ;;
    esac
}

curl_common() {
    curl --proto '=https' --proto-redir '=https' --location --fail --silent --show-error \
        --connect-timeout 10 --max-time 120 --retry 2 --retry-delay 1 "$@"
}

resolve_latest_public() {
    effective=$(curl_common --head --output /dev/null --write-out '%{url_effective}' "$REPO_URL/releases/latest") ||
        fail "could not resolve the latest release from $REPO_URL/releases/latest"
    prefix="$REPO_URL/releases/tag/"
    case "$effective" in
        "$prefix"*) TAG=${effective#"$prefix"} ;;
        *) fail "latest release redirected outside the expected repository: $effective" ;;
    esac
    is_valid_tag "$TAG" || fail "latest release returned an invalid tag: $TAG"
    [ "$effective" = "$prefix$TAG" ] || fail "latest release returned an unexpected URL: $effective"
}

acquire_release() {
    require_command curl
    if [ -z "$TAG" ]; then
        resolve_latest_public
    fi

    VERSION=${TAG#v}
    ROOT="openjev-${TAG}-${TARGET}"
    ASSET="$ROOT.tar.gz"
    ARCHIVE="$TEMP_DIR/$ASSET"
    CHECKSUMS="$TEMP_DIR/SHA256SUMS"
    base="$REPO_URL/releases/download/$TAG"

    curl_common --output "$ARCHIVE" "$base/$ASSET" ||
        fail "could not download $ASSET from $base"
    curl_common --output "$CHECKSUMS" "$base/SHA256SUMS" ||
        fail "could not download SHA256SUMS from $base"
    [ -f "$ARCHIVE" ] && [ ! -L "$ARCHIVE" ] || fail "release archive was not downloaded"
    [ -f "$CHECKSUMS" ] && [ ! -L "$CHECKSUMS" ] || fail "SHA256SUMS was not downloaded"
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        fail "sha256sum or shasum is required to verify the release"
    fi
}

verify_checksum() {
    hashes="$TEMP_DIR/selected-hashes"
    awk -v wanted="$ASSET" '
        {
            for (field = 1; field <= NF; field++) {
                if ($field == wanted || $field == "*" wanted) {
                    if (NF == 2 && field == 2) print $1
                    else print "INVALID"
                    next
                }
            }
        }
    ' "$CHECKSUMS" >"$hashes" || fail "could not read SHA256SUMS"
    count=$(awk 'END { print NR + 0 }' "$hashes")
    [ "$count" -eq 1 ] || fail "SHA256SUMS must contain exactly one entry for $ASSET"
    expected=$(awk 'NR == 1 { print tolower($0) }' "$hashes")
    [ "${#expected}" -eq 64 ] || fail "invalid SHA-256 for $ASSET"
    case "$expected" in
        *[!0-9A-Fa-f]*) fail "invalid SHA-256 for $ASSET" ;;
    esac
    actual=$(sha256_file "$ARCHIVE") || fail "could not hash $ASSET"
    [ "$actual" = "$expected" ] || fail "SHA-256 mismatch for $ASSET"
}

verify_archive() {
    names="$TEMP_DIR/archive-names"
    verbose="$TEMP_DIR/archive-verbose"
    tar -tzf "$ARCHIVE" >"$names" 2>"$TEMP_DIR/tar-list-error" || fail "invalid release archive: $ASSET"
    tar -tvzf "$ARCHIVE" >"$verbose" 2>"$TEMP_DIR/tar-verbose-error" || fail "invalid release archive metadata: $ASSET"

    awk -v root="$ROOT" '
        BEGIN {
            expected[root] = "d"
            expected[root "/openjev"] = "f"
            expected[root "/LICENSE"] = "f"
            expected[root "/THIRD_PARTY.md"] = "f"
            expected[root "/THIRD_PARTY_LICENSES.html"] = "f"
            expected[root "/RUST-COPYRIGHT-library.html"] = "f"
            expected[root "/colored-3.1.1.crate"] = "f"
            expected[root "/option-ext-0.2.0.crate"] = "f"
            expected[root "/README.md"] = "f"
            expected[root "/SERVE.md"] = "f"
            expected[root "/BUILD-INFO.json"] = "f"
        }
        {
            raw = $0; name = raw; sub(/\/$/, "", name)
            if (raw ~ /\\/ || name ~ /^\// || name !~ /^[A-Za-z0-9._+\/-]+$/ ||
                name ~ /(^|\/)\.\.?($|\/)/ || !(name in expected) || seen[name]++) bad = 1
            count++
        }
        END { if (bad || count != 11) exit 1 }
    ' "$names" || fail "archive members do not match the exact required release payload"

    awk -v root="$ROOT" '
        BEGIN {
            expected[root] = "d"
            expected[root "/openjev"] = "-"
            expected[root "/LICENSE"] = "-"
            expected[root "/THIRD_PARTY.md"] = "-"
            expected[root "/THIRD_PARTY_LICENSES.html"] = "-"
            expected[root "/RUST-COPYRIGHT-library.html"] = "-"
            expected[root "/colored-3.1.1.crate"] = "-"
            expected[root "/option-ext-0.2.0.crate"] = "-"
            expected[root "/README.md"] = "-"
            expected[root "/SERVE.md"] = "-"
            expected[root "/BUILD-INFO.json"] = "-"
        }
        {
            name = $NF; sub(/\/$/, "", name); type = substr($1, 1, 1)
            if (!(name in expected) || type != expected[name] || seen[name]++) bad = 1
            count++
        }
        END { if (bad || count != 11) exit 1 }
    ' "$verbose" || fail "archive contains a link, device, unexpected directory, or non-regular payload member"
}

extract_and_check() {
    INSTALL_STAGE=$(mktemp -d "$BIN_DIR/.install.XXXXXX") || fail "could not create private install staging directory"
    chmod 700 "$INSTALL_STAGE" || fail "could not secure install staging directory"
    tar -xzf "$ARCHIVE" -C "$INSTALL_STAGE" > /dev/null 2>"$TEMP_DIR/tar-extract-error" || fail "could not extract $ASSET"
    STAGED_ROOT="$INSTALL_STAGE/$ROOT"
    [ -d "$STAGED_ROOT" ] && [ ! -L "$STAGED_ROOT" ] || fail "archive did not extract the expected root directory"
    for member in $MEMBER_FILES; do
        path="$STAGED_ROOT/$member"
        [ -f "$path" ] && [ ! -L "$path" ] || fail "archive member is not an owned regular file: $member"
        if [ "$member" = openjev ]; then
            chmod 755 "$path" || fail "could not set executable permissions"
        else
            chmod 644 "$path" || fail "could not set safe permissions on $member"
        fi
    done
    chmod 755 "$STAGED_ROOT" || fail "could not set payload directory permissions"
}

clear_macos_quarantine() {
    binary=$1
    [ "$TARGET" = aarch64-apple-darwin ] || return 0
    attrs="$TEMP_DIR/xattrs"
    if ! xattr "$binary" >"$attrs" 2>"$TEMP_DIR/xattr-list-error"; then
        fail "could not inspect quarantine attributes on the verified executable"
    fi
    if grep -Fx 'com.apple.quarantine' "$attrs" >/dev/null 2>&1; then
        xattr -d com.apple.quarantine "$binary" >/dev/null 2>"$TEMP_DIR/xattr-delete-error" ||
            fail "could not remove com.apple.quarantine from the verified executable"
    fi
}

verify_binary_version() {
    binary=$1
    output="$TEMP_DIR/version-output"
    errors="$TEMP_DIR/version-errors"
    if ! (CDPATH=; cd "$TEMP_DIR" && "$binary" --version >"$output" 2>"$errors"); then
        fail "verified release executable failed its --version check"
    fi
    [ ! -s "$errors" ] || fail "verified release executable wrote to stderr during --version"
    [ "$(awk 'END { print NR + 0 }' "$output")" -eq 1 ] || fail "verified release executable emitted invalid version JSON"
    version_json=$(awk 'NR == 1 { print }' "$output")
    prefix="{\"schema\":\"openjev-version-v1\",\"version\":\"$VERSION\",\"build\":\""
    case "$version_json" in
        "$prefix"*) build_and_end=${version_json#"$prefix"} ;;
        *) fail "verified release executable reported the wrong schema or version" ;;
    esac
    build=${build_and_end%\"\}}
    [ "$build_and_end" = "${build}\"}" ] || fail "verified release executable emitted invalid version JSON"
    case "$build" in
        ''|*[!0-9A-Za-z._+-]*) fail "verified release executable emitted invalid version JSON" ;;
    esac
}

compare_or_install_payload() {
    PAYLOAD="$BIN_DIR/$ROOT"
    if [ -L "$PAYLOAD" ]; then
        fail "refusing symlinked release payload: $PAYLOAD"
    fi
    if [ -e "$PAYLOAD" ]; then
        [ -d "$PAYLOAD" ] || fail "existing release payload is not a directory: $PAYLOAD"
        # Filenames in an installer-owned release payload are a closed ASCII set.
        # shellcheck disable=SC2012
        entry_count=$(ls -A "$PAYLOAD" 2>/dev/null | awk 'END { print NR + 0 }') || fail "could not inspect existing release payload"
        [ "$entry_count" -eq 10 ] || fail "existing same-version payload is not byte-identical"
        for member in $MEMBER_FILES; do
            existing="$PAYLOAD/$member"
            [ -f "$existing" ] && [ ! -L "$existing" ] || fail "existing same-version payload is not byte-identical"
            cmp -s "$STAGED_ROOT/$member" "$existing" || fail "existing same-version payload differs: $member"
        done
        reused_binary="$PAYLOAD/openjev"
        chmod 755 "$reused_binary" || fail "could not set executable permissions on reused payload"
        clear_macos_quarantine "$reused_binary"
        verify_binary_version "$reused_binary"
        rm -rf "$INSTALL_STAGE"
        INSTALL_STAGE=""
    else
        mv "$STAGED_ROOT" "$PAYLOAD" || fail "could not install verified release payload"
        rmdir "$INSTALL_STAGE" 2>/dev/null || :
        INSTALL_STAGE=""
    fi
}

atomic_link() {
    target=$1
    destination=$2
    directory=${destination%/*}
    LINK_STAGE=$(mktemp -d "$directory/.openjev-link.XXXXXX") || fail "could not stage link for $destination"
    chmod 700 "$LINK_STAGE" || fail "could not secure link staging directory"
    ln -s "$target" "$LINK_STAGE/openjev" || fail "could not create staged link for $destination"
    mv -f "$LINK_STAGE/openjev" "$destination" || fail "could not atomically activate $destination"
    rmdir "$LINK_STAGE" || fail "could not remove link staging directory"
    LINK_STAGE=""
}

activate_installation() {
    atomic_link "$PAYLOAD/openjev" "$ACTIVE_LINK"
    if [ "$REPLACE_COMMAND_LINK" -eq 1 ]; then
        atomic_link "$ACTIVE_LINK" "$COMMAND_LINK"
    fi
}

print_path_hint() {
    case ":${PATH:-}:" in
        *":$LINK_DIR:"*) ;;
        *)
            say "openjev installer: $LINK_DIR is not on PATH"
            say "openjev installer: add this to your shell profile: export PATH=\"$LINK_DIR:\$PATH\""
            ;;
    esac
}

main() {
    TAG=""
    case $# in
        0) ;;
        1)
            [ "$1" = "--help" ] || fail "unknown argument: $1"
            usage
            return 0
            ;;
        2)
            [ "$1" = "--version" ] || fail "unknown argument: $1"
            is_valid_tag "$2" || fail "--version must be a tag like v0.1.0"
            TAG=$2
            ;;
        *) fail "usage: install.sh [--version vMAJOR.MINOR.PATCH]" ;;
    esac

    umask 077
    require_command grep
    require_command awk
    require_command uname
    require_command mkdir
    require_command mktemp
    require_command tar
    require_command chmod
    require_command mv
    require_command ln
    require_command readlink
    require_command cmp
    if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
        fail "sha256sum or shasum is required to verify the release"
    fi
    check_home
    detect_target
    trap cleanup 0
    trap on_signal 1 2 15
    prepare_directories
    check_existing_links

    TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/openjev-install.XXXXXX") || fail "could not create private temporary directory"
    chmod 700 "$TEMP_DIR" || fail "could not secure temporary directory"
    acquire_release
    verify_checksum
    verify_archive
    extract_and_check
    clear_macos_quarantine "$STAGED_ROOT/openjev"
    verify_binary_version "$STAGED_ROOT/openjev"
    compare_or_install_payload
    activate_installation
    print_path_hint
    say "openjev installer: installed $TAG for $TARGET"
}

main "$@"
