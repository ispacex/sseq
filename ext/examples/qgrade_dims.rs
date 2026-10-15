//! Resolves an unstable module and prints `n,s,e,num_gens` where e is the Q-weight
//! (number of Milnor Q_i's) of each generator, read off from its differential.
//! Checks that every differential is homogeneous in Q-weight; panics otherwise.
//!
//! At odd p the Steenrod algebra is graded by the number of Q's in the Milnor basis (the dual
//! A_* is graded by the number of tau's, and the coproduct keeps it), so a minimal resolution
//! built by row reduction is automatically weight-homogeneous: Ext^{s,t} splits by e. Here
//! `n = t - s`, `e` = weight of the generator. Requires the Milnor basis (the default).
//!
//! Example: `printf 'S_3[3]\n\n150\n60\n' | target/release/examples/qgrade_dims`

use std::collections::HashMap;

use algebra::MilnorAlgebra;
use ext::chain_complex::{ChainComplex, FreeChainComplex};
use sseq::coordinates::Bidegree;

fn main() -> anyhow::Result<()> {
    ext::utils::init_logging()?;
    let res = ext::utils::query_unstable_module(false)?;
    let max = Bidegree::n_s(
        query::raw("Max n", str::parse),
        query::raw("Max s", str::parse),
    );
    res.compute_through_stem(max);

    let alg = res.algebra();
    let milnor: &MilnorAlgebra = (&*alg).try_into().expect("need Milnor basis");

    // weight[(s, t, idx)]
    let mut weight: HashMap<(i32, i32, usize), u32> = HashMap::new();
    let mut bidegs: Vec<Bidegree> = res.iter_stem().collect();
    bidegs.sort_by_key(|b| (b.s(), b.t()));
    let mut terms_checked: u64 = 0;
    let mut out: Vec<(i32, i32, u32, usize)> = Vec::new();
    for b in bidegs {
        let ngens = res.number_of_gens_in_bidegree(b);
        let mut counts: HashMap<u32, usize> = HashMap::new();
        for i in 0..ngens {
            let w = if b.s() == 0 {
                0
            } else {
                let d = res.differential(b.s());
                let dx = d.output(b.t(), i);
                let tgt = res.module(b.s() - 1);
                let mut w: Option<u32> = None;
                for (idx, _c) in dx.iter_nonzero() {
                    let og = tgt.index_to_op_gen(b.t(), idx);
                    let m = milnor.basis_element_from_index(og.operation_degree, og.operation_index);
                    let gw = weight[&(b.s() - 1, og.generator_degree, og.generator_index)];
                    let tw = gw + m.q_part.count_ones();
                    terms_checked += 1;
                    match w {
                        None => w = Some(tw),
                        Some(x) => assert_eq!(x, tw, "inhomogeneous d at {b} gen {i}"),
                    }
                }
                w.expect("zero differential")
            };
            weight.insert((b.s(), b.t(), i), w);
            *counts.entry(w).or_default() += 1;
        }
        for (e, c) in counts {
            out.push((b.n(), b.s(), e, c));
        }
    }
    out.sort();
    for (n, s, e, c) in out {
        println!("{n},{s},{e},{c}");
    }
    eprintln!("terms checked: {terms_checked}, all homogeneous");
    Ok(())
}
