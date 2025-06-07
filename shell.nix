with import <nixpkgs> {};
pkgs.mkShell {
  buildInputs = [
    python3
    python3Packages.pip
    python3Packages.streamlit
    python3Packages.numpy
    python3Packages.pandas
    # add other python libs
  ];
  shellHook = ''
    # add environment variable
    echo "Python ENV"
  '';
}