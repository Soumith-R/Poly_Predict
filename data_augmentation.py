import random
from rdkit import Chem

class PolymerDataAugmentor:
    def __init__(self, substitution_dict=None):
        """
        Initialize the data augmentor with a substitution dictionary.
        
        Args:
            substitution_dict: A dictionary mapping atoms/groups to their substitutes.
        """
        self.substitution_dict = substitution_dict or {
            'C': ['N', 'O'],  # Example substitutions
            'N': ['C', 'S'],
            # Add more substitutions as needed
        }

    def random_substitution(self, smiles: str) -> str:
        """
        Perform random substitution on a SMILES string.
        
        Args:
            smiles: Input SMILES string
            
        Returns:
            Augmented SMILES string
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles  # Return original if invalid SMILES
        
        # Get all atoms in the molecule
        atoms = mol.GetAtoms()
        atom_indices = list(range(len(atoms)))
        
        # Randomly select an atom to substitute
        if atom_indices:
            idx_to_substitute = random.choice(atom_indices)
            atom = atoms[idx_to_substitute]
            atom_symbol = atom.GetSymbol()
            
            # Check if substitution is possible
            if atom_symbol in self.substitution_dict:
                substitute = random.choice(self.substitution_dict[atom_symbol])
                atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(substitute))
            else:
                print(f"No substitution available for {atom_symbol}.")  # Log or handle this case
        
        return Chem.MolToSmiles(mol)

    def augment(self, smiles_list: list) -> list:
        """
        Augment a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of augmented SMILES strings
        """
        augmented_smiles = []
        for smiles in smiles_list:
            augmented_smiles.append(self.random_substitution(smiles))
        return augmented_smiles