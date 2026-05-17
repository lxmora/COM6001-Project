{
  description = "Basic C++ development flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { nixpkgs, ... } @ inputs: 
  let
    pkgs = nixpkgs.legacyPackages.x86_64-linux;
  in
  { 
    devShells.x86_64-linux.default = pkgs.mkShell {
        buildInputs = with pkgs; [ 
          python313
	  python313Packages.networkx
	  python313Packages.matplotlib
	  python313Packages.scikit-learn
	  python313Packages.scipy
	  python313Packages.numpy
          ];
      };
  };
}
