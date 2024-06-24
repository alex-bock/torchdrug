
import pandas as pd

import torch
from torch import Tensor

from torchdrug.data import Protein


CONTACT2ID = {"hb": 4, "sb": 5, "pc": 6, "ps": 7, "ts": 8, "hp": 9, "vdw": 10}


def edit_contact_type(contact_type: str):

    if contact_type.startswith("hb"):
        contact_type = contact_type[:2]
    elif contact_type.startswith("hp"):
        contact_type = "vdw"
    elif contact_type.startswith("vdw"):
        contact_type = "hp"

    return contact_type


def load_contacts(protein: Protein, contacts_fp: str):

    contacts_df = pd.read_csv(
        contacts_fp, sep="\t", names=["frame", "type", "u", "v"],
        skiprows=[0, 1], usecols=[0, 1, 2, 3]
    )

    edge_list = protein.edge_list

    contacts_df.type = contacts_df.type.apply(edit_contact_type)

    contacts_df[["chain_u", "res_type_u", "chain_u_i", "atom_name_u"]] = contacts_df.u.str.split(":", expand=True)
    contacts_df[["chain_v", "res_type_v", "chain_v_i", "atom_name_v"]] = contacts_df.v.str.split(":", expand=True)
    # import pdb; pdb.set_trace()

    rstrip_fn = lambda x: x[:-1] if x[-1].isalpha() else x
    contacts_df["chain_u_i"] = contacts_df.chain_u_i.apply(rstrip_fn)
    contacts_df["chain_v_i"] = contacts_df.chain_v_i.apply(rstrip_fn)
    # import pdb; pdb.set_trace()

    contacts_df["res_u_i"] = contacts_df.apply(lambda row: int((protein.chain_id == protein.alphabet2id[row.chain_u]).nonzero()[int(row.chain_u_i) - 1]), axis=1)
    contacts_df["res_v_i"] = contacts_df.apply(lambda row: int((protein.chain_id == protein.alphabet2id[row.chain_v]).nonzero()[int(row.chain_v_i) - 1]), axis=1)
    # import pdb; pdb.set_trace()

    contacts_df["res_u_atom_is"] = contacts_df.apply(lambda row: (protein.atom2residue == row.res_u_i).nonzero().squeeze(), axis=1)
    contacts_df["res_v_atom_is"] = contacts_df.apply(lambda row: (protein.atom2residue == row.res_v_i).nonzero().squeeze(), axis=1)
    # import pdb; pdb.set_trace()

    contacts_df["res_u_atom_names"] = contacts_df.res_u_atom_is.apply(lambda x: protein.atom_name[x])
    contacts_df["res_v_atom_names"] = contacts_df.res_v_atom_is.apply(lambda x: protein.atom_name[x])
    # import pdb; pdb.set_trace()

    contacts_df["atom_u_i"] = contacts_df.apply(lambda row: row.res_u_atom_is[row.res_u_atom_names == protein.atom_name2id[row.atom_name_u]][0], axis=1)
    contacts_df["atom_v_i"] = contacts_df.apply(lambda row: row.res_v_atom_is[row.res_v_atom_names == protein.atom_name2id[row.atom_name_v]][0], axis=1)

    # import pdb; pdb.set_trace()

    contacts_df["contact_id"] = contacts_df.type.apply(lambda x: CONTACT2ID[x])
    # import pdb; pdb.set_trace()

    edge_list = torch.cat(
        [
            edge_list,
            torch.stack(
                [
                    Tensor(contacts_df.atom_u_i.values.astype(int)),
                    Tensor(contacts_df.atom_v_i.values.astype(int)),
                    Tensor(contacts_df.contact_id.values)
                ]
            ).t().int()
        ]
    )

    try:
        return Protein(
            edge_list, atom_type=protein.atom_type, bond_type=edge_list[:, 2],
            residue_type=protein.residue_type, view=protein.view,
            atom_name=protein.atom_name, atom2residue=protein.atom2residue,
            residue_feature=protein.residue_feature, num_node=len(protein.atom_type),
            is_hetero_atom=protein.is_hetero_atom, occupancy=protein.occupancy,
            b_factor=protein.b_factor, residue_number=protein.residue_number,
            insertion_code=protein.insertion_code, chain_id=protein.chain_id,
            num_relation=edge_list[:, 2].max() + 1,
            node_position=protein.node_position

        )
    except ValueError:
        import pdb; pdb.set_trace()