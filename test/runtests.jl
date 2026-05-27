using Test
using Dictionaries
using LinearAlgebra
using VectorInterface
using TensorKit
using MatrixAlgebraKit
using ChainRulesCore
using Zygote
using FiniteDifferences
using Random
using FragmentedTensors
using FragmentedTensors: SpaceID, FragmentedTensor, NullSpaceID, assemble_global, disassemble_global, disassemble_global_U, disassemble_global_V

# Define elementwise dictionary conversion for cross-validation
function FragmentedTensor_elementwise(dict::Dictionary{SpaceID{N}, TensorType}) where {N, TensorType}
    new_dict = Dictionary{Tuple{SpaceID{N}, SpaceID{0}}, TensorType}()
    for (k, v) in pairs(dict)
        insert!(new_dict, (k, NullSpaceID), v)
    end
    return FragmentedTensor{N, 0, TensorType}(new_dict)
end

@testset "FragmentedTensors and SpaceID" begin

    @testset "SpaceID Logic" begin
        # 1. Parsing & macros
        s1 = sid"A, B', C"
        @test s1.labels == ("A", "B", "C")
        @test s1.adjoint == (false, true, false)
        
        s2 = SpaceID(("A", "B", "C"), (false, true, false))
        @test s1 == s2
        @test hash(s1) == hash(s2)
        
        s3 = sid"A, B"
        @test s1 != s3
        @test s1 != sid"A, B, C'"
        
        # Empty space / NullSpaceID
        s_empty = sid""
        @test s_empty == NullSpaceID
        @test length(s_empty.labels) == 0
        @test length(s_empty.adjoint) == 0
        
        # 2. String formatting
        @test string(s1) == "A, B', C"
        @test string(NullSpaceID) == ""
        
        # 3. Adjoint operator
        s1_adj = s1'
        @test s1_adj.labels == ("A", "B", "C")
        @test s1_adj.adjoint == (true, false, true)
        @test (s1_adj)' == s1
        @test NullSpaceID' == NullSpaceID
        
        # 4. Multiplication (concatenation)
        s_mult = s1 * s3
        @test s_mult.labels == ("A", "B", "C", "A", "B")
        @test s_mult.adjoint == (false, true, false, false, false)
    end

    @testset "FragmentedTensor Struct & Indexing" begin
        V = ℂ^2
        # Dictionary keyed by single SpaceID
        d_single = Dictionary([sid"A", sid"B"], [randn(ComplexF64, V, V), randn(ComplexF64, V, V)])
        
        # Test our optimized order-preserving constructor
        FT_opt = FragmentedTensor(d_single)
        FT_ew = FragmentedTensor_elementwise(d_single)
        
        # Cross-validation
        @test keys(FT_opt.data) == keys(FT_ew.data)
        @test values(FT_opt.data) == values(FT_ew.data)
        
        # N_out = 0 and N_in = 0 edge cases
        FT_00 = FragmentedTensor{0, 0, Matrix{Float64}}()
        @test isempty(FT_00.data)
        
        # Dict lookup using FT[(S_out, S_in)]
        S_out = sid"A"
        S_in = sid"B"
        t_val = randn(ComplexF64, V, V)
        ft_pairs = FragmentedTensor(Dictionary([(S_out, S_in)], [t_val]))
        @test ft_pairs[(S_out, S_in)] === t_val
        @test haskey(ft_pairs, (S_out, S_in))
        
        # Dict lookup using FT[S] where S = S_out * S_in'
        S_combined = S_out * S_in'
        @test ft_pairs[S_combined] === t_val
        @test haskey(ft_pairs, S_combined)
        
        # Mutating via setindex!
        t_val_new = randn(ComplexF64, V, V)
        ft_pairs[S_combined] = t_val_new
        @test ft_pairs[(S_out, S_in)] === t_val_new
        
        ft_pairs[(S_out, S_in)] = t_val
        @test ft_pairs[S_combined] === t_val

        # Testing setting a completely new key (requires set! to avoid IndexError)
        t_val_brandnew = randn(ComplexF64, V, V)
        S_new_combined = sid"X" * (sid"Y")'
        ft_pairs[S_new_combined] = t_val_brandnew
        @test ft_pairs[(sid"X", sid"Y")] === t_val_brandnew
    end

    @testset "Basic Linear Algebra" begin
        # 1. Sum and difference
        V = ℂ^2
        FT1 = FragmentedTensor(Dictionary([(sid"A", sid"B"), (sid"C", sid"D")], [1.0, 2.0]))
        FT2 = FragmentedTensor(Dictionary([(sid"C", sid"D"), (sid"E", sid"F")], [3.0, 4.0]))
        
        FT_sum = FT1 + FT2
        @test FT_sum[(sid"A", sid"B")] == 1.0
        @test FT_sum[(sid"C", sid"D")] == 5.0
        @test FT_sum[(sid"E", sid"F")] == 4.0
        
        FT_diff = FT1 - FT2
        @test FT_diff[(sid"A", sid"B")] == 1.0
        @test FT_diff[(sid"C", sid"D")] == -1.0
        @test FT_diff[(sid"E", sid"F")] == -4.0
        
        # 2. Scalar mult and division (Broadcasting)
        FT_mult1 = FT1 * 3.0
        FT_mult2 = 3.0 * FT1
        FT_div = FT1 / 2.0
        
        @test FT_mult1[(sid"A", sid"B")] == 3.0
        @test FT_mult2[(sid"A", sid"B")] == 3.0
        @test FT_div[(sid"A", sid"B")] == 0.5
        
        # 3. Norm
        @test norm(FT1) ≈ sqrt(1.0^2 + 2.0^2)
        
        # 4. Dot products
        # Fragmented tensors live in a single direct-sum space; missing keys
        # are true zeros, so dot sums over the *intersection* of keysets.
        # FT1 keys: {(A,B), (C,D)}, FT2 keys: {(C,D), (E,F)} -> only (C,D)
        # contributes: dot(2.0, 3.0) = 6.0.
        @test dot(FT1, FT2) ≈ 2.0 * 3.0

        # Key sets identical -> sum key-by-key
        FT3 = FragmentedTensor(Dictionary([(sid"A", sid"B"), (sid"C", sid"D")], [3.0, 4.0]))
        @test dot(FT1, FT3) ≈ 1.0*3.0 + 2.0*4.0

        # Disjoint keysets -> empty intersection -> 0
        FT_disjoint = FragmentedTensor(Dictionary([(sid"E", sid"F")], [5.0]))
        @test dot(FT1, FT_disjoint) == 0.0

        # Empty cases
        FT_empty = FragmentedTensor{1, 1, Float64}()
        @test dot(FT_empty, FT_empty) == 0.0
        @test dot(FT1, FT_empty) == 0.0
    end

    @testset "Matrix Multiplication" begin
        # Matrices for multiplication (simplest non-tensor type)
        # N_out = 1, N_in = 1
        # inconsistent N_in and N_out should throw
        FT_A = FragmentedTensor(Dictionary([(sid"i", sid"j")], [fill(1.0, 2, 2)]))
        FT_B_bad = FragmentedTensor{2, 1, Matrix{Float64}}() # N_out=2, N_in=1
        @test_throws DimensionMismatch FT_A * FT_B_bad
        
        # Consistent but no matching keys -> empty dictionary
        FT_B_nomatch = FragmentedTensor(Dictionary([(sid"x", sid"y")], [fill(2.0, 2, 2)]))
        FT_C_nomatch = FT_A * FT_B_nomatch
        @test isempty(FT_C_nomatch.data)
        
        # Partial matching and multiple matches summed up (3 fragments per tensor)
        # We define keys:
        # FT1 has: (sid"a", sid"b"), (sid"a", sid"c"), (sid"x", sid"y")
        # FT2 has: (sid"b", sid"d"), (sid"c", sid"d"), (sid"y", sid"z")
        # Matches:
        # 1. (sid"a", sid"b") * (sid"b", sid"d") -> under key (sid"a", sid"d")
        # 2. (sid"a", sid"c") * (sid"c", sid"d") -> under key (sid"a", sid"d")  <-- Should sum up!
        # 3. (sid"x", sid"y") * (sid"y", sid"z") -> under key (sid"x", sid"z")
        
        m_a_b = [1.0 2.0; 3.0 4.0]
        m_a_c = [5.0 6.0; 7.0 8.0]
        m_x_y = [0.1 0.2; 0.3 0.4]
        
        m_b_d = [2.0 0.0; 0.0 2.0]
        m_c_d = [3.0 0.0; 0.0 3.0]
        m_y_z = [10.0 20.0; 30.0 40.0]
        
        FT1 = FragmentedTensor(Dictionary([(sid"a", sid"b"), (sid"a", sid"c"), (sid"x", sid"y")], [m_a_b, m_a_c, m_x_y]))
        FT2 = FragmentedTensor(Dictionary([(sid"b", sid"d"), (sid"c", sid"d"), (sid"y", sid"z")], [m_b_d, m_c_d, m_y_z]))
        
        FT3 = FT1 * FT2
        @test length(FT3.data) == 2
        @test haskey(FT3, (sid"a", sid"d"))
        @test haskey(FT3, (sid"x", sid"z"))
        
        # Cross-verify manual calculation
        expected_ad = m_a_b * m_b_d + m_a_c * m_c_d
        expected_xz = m_x_y * m_y_z
        
        @test FT3[(sid"a", sid"d")] ≈ expected_ad
        @test FT3[(sid"x", sid"z")] ≈ expected_xz

        # Test 5: Varied and longer SpaceIDs (N_out=3, N_in=2 multiplied by N_out=2, N_in=4)
        m_left = [1.0 2.0; 3.0 4.0]
        m_right = [2.0 0.0 1.0 0.0; 0.0 2.0 0.0 1.0]
        
        FT_long1 = FragmentedTensor(Dictionary([(sid"a, b, c", sid"d, e")], [m_left]))
        FT_long2 = FragmentedTensor(Dictionary([(sid"d, e", sid"f, g, h, i")], [m_right]))
        
        FT_long_res = FT_long1 * FT_long2
        @test typeof(FT_long_res) === FragmentedTensor{3, 4, Matrix{Float64}}
        @test haskey(FT_long_res, (sid"a, b, c", sid"f, g, h, i"))
        @test FT_long_res[(sid"a, b, c", sid"f, g, h, i")] ≈ m_left * m_right
    end

    @testset "VectorInterface Compatibility" begin
        FT1 = FragmentedTensor(Dictionary([(sid"A", sid"B"), (sid"C", sid"D")], [fill(1.0, 1, 1), fill(2.0, 1, 1)]))
        FT2 = FragmentedTensor(Dictionary([(sid"C", sid"D"), (sid"E", sid"F")], [fill(3.0, 1, 1), fill(4.0, 1, 1)]))
        
        @test scalartype(FT1) == Float64
        
        # zerovector
        FT_zero = zerovector(FT1, Float64)
        @test isempty(FT_zero.data)
        
        # scale
        FT_scaled = scale(FT1, 5.0)
        @test FT_scaled[(sid"A", sid"B")] == fill(5.0, 1, 1)
        
        # scale!
        FT1_copy = deepcopy(FT1)
        scale!(FT1_copy, 5.0)
        @test FT1_copy[(sid"A", sid"B")] == fill(5.0, 1, 1)
        
        # scale!!
        scale!!(FT1_copy, 2.0)
        @test FT1_copy[(sid"A", sid"B")] == fill(10.0, 1, 1)
        
        # 3-argument scale, scale!, scale!!
        FT_scaled_3 = scale(FT2, FT1, 5.0)
        @test FT_scaled_3[(sid"A", sid"B")] == fill(5.0, 1, 1)
        @test !haskey(FT_scaled_3, (sid"E", sid"F"))
        
        FT_copy_3 = deepcopy(FT2)
        scale!(FT_copy_3, FT1, 5.0)
        @test FT_copy_3[(sid"A", sid"B")] == fill(5.0, 1, 1)
        @test !haskey(FT_copy_3, (sid"E", sid"F"))
        
        FT_copy_3_b = deepcopy(FT2)
        scale!!(FT_copy_3_b, FT1, 2.0)
        @test FT_copy_3_b[(sid"A", sid"B")] == fill(2.0, 1, 1)
        @test !haskey(FT_copy_3_b, (sid"E", sid"F"))
        
        # add
        FT_add = add(FT1, FT2, 2.0, 3.0)
        @test FT_add[(sid"A", sid"B")] == fill(3.0, 1, 1)
        @test FT_add[(sid"C", sid"D")] == fill(12.0, 1, 1)
        @test FT_add[(sid"E", sid"F")] == fill(8.0, 1, 1)
        
        # add!
        FT1_copy2 = deepcopy(FT1)
        add!(FT1_copy2, FT2, 2.0, 3.0)
        @test FT1_copy2[(sid"A", sid"B")] == fill(3.0, 1, 1)
        @test FT1_copy2[(sid"C", sid"D")] == fill(12.0, 1, 1)
        @test FT1_copy2[(sid"E", sid"F")] == fill(8.0, 1, 1)
        
        # inner
        FT_dot1 = FragmentedTensor(Dictionary([(sid"A", sid"B")], [fill(2.0, 1, 1)]))
        FT_dot2 = FragmentedTensor(Dictionary([(sid"A", sid"B")], [fill(3.0, 1, 1)]))
        @test inner(FT_dot1, FT_dot2) == 6.0
    end

    @testset "Diverse Spaces (TensorKit)" begin
        # 1. ComplexSpace
        V_c = ℂ^2
        t1_c = randn(ComplexF64, V_c, V_c)
        t2_c = randn(ComplexF64, V_c, V_c)
        FT_A_c = FragmentedTensor(Dictionary([(sid"i", sid"j")], [t1_c]))
        FT_B_c = FragmentedTensor(Dictionary([(sid"j", sid"k")], [t2_c]))
        FT_C_c = FT_A_c * FT_B_c
        @test FT_C_c[(sid"i", sid"k")] ≈ t1_c * t2_c

        # 2. Z2Space
        V_z2 = Z2Space(0=>2, 1=>2)
        t1_z2 = randn(ComplexF64, V_z2, V_z2)
        t2_z2 = randn(ComplexF64, V_z2, V_z2)
        FT_A_z2 = FragmentedTensor(Dictionary([(sid"i", sid"j")], [t1_z2]))
        FT_B_z2 = FragmentedTensor(Dictionary([(sid"j", sid"k")], [t2_z2]))
        FT_C_z2 = FT_A_z2 * FT_B_z2
        @test FT_C_z2[(sid"i", sid"k")] ≈ t1_z2 * t2_z2

        # 3. Z3Space
        V_z3 = Z3Space(0=>1, 1=>1, 2=>1)
        t1_z3 = randn(ComplexF64, V_z3, V_z3)
        t2_z3 = randn(ComplexF64, V_z3, V_z3)
        FT_A_z3 = FragmentedTensor(Dictionary([(sid"i", sid"j")], [t1_z3]))
        FT_B_z3 = FragmentedTensor(Dictionary([(sid"j", sid"k")], [t2_z3]))
        FT_C_z3 = FT_A_z3 * FT_B_z3
        @test FT_C_z3[(sid"i", sid"k")] ≈ t1_z3 * t2_z3

        # 4. U1Space
        V_u1 = U1Space(0=>2, 1=>2)
        t1_u1 = randn(ComplexF64, V_u1, V_u1)
        t2_u1 = randn(ComplexF64, V_u1, V_u1)
        FT_A_u1 = FragmentedTensor(Dictionary([(sid"i", sid"j")], [t1_u1]))
        FT_B_u1 = FragmentedTensor(Dictionary([(sid"j", sid"k")], [t2_u1]))
        FT_C_u1 = FT_A_u1 * FT_B_u1
        @test FT_C_u1[(sid"i", sid"k")] ≈ t1_u1 * t2_u1

        # 5. SU2Space
        V_su2 = SU2Space(0=>2, 1/2=>1)
        t1_su2 = randn(ComplexF64, V_su2, V_su2)
        t2_su2 = randn(ComplexF64, V_su2, V_su2)
        FT_A_su2 = FragmentedTensor(Dictionary([(sid"i", sid"j")], [t1_su2]))
        FT_B_su2 = FragmentedTensor(Dictionary([(sid"j", sid"k")], [t2_su2]))
        FT_C_su2 = FT_A_su2 * FT_B_su2
        @test FT_C_su2[(sid"i", sid"k")] ≈ t1_su2 * t2_su2
    end

    @testset "Fragmented Factorizations & AD" begin
        # =====================================================================
        # Test design (plan v3):
        #
        # - Each scenario is a (space_dict, keys, is_hermitian) triple defining
        #   the *structural* family of a FragmentedTensor. The scenario does
        #   not fix a specific random matrix.
        # - We *fuzz* every scenario: run `N_REPS` random repetitions, each
        #   generating a fresh `A`, perturbation `Δ` and weight `M`, then
        #   verify reconstruction and AD-vs-FD for every applicable
        #   decomposition (eigh / eig / svd, plus truncated variants — TODO).
        # - Per-repetition outcomes (pass / excused / real-fail) are aggregated;
        #   the scenario passes iff there are no real failures AND at least one
        #   repetition was not excused.
        # - Excuses:
        #     · "ill_cond"      — only for `eig_full` (uses `inv(U)`); excused
        #                         when `cond(U) > ILL_COND_THRESHOLD`.
        #     · "near_boundary" — only for truncated decompositions; excused
        #                         when the gap between last-kept and
        #                         first-truncated eigenvalue is small.
        # - One @testset per scenario (rep count is part of the name).
        # =====================================================================

        # ---- helpers ----

        function get_product_space(sid::SpaceID, space_dict)
            S_type = spacetype(typeof(first(values(space_dict))))
            # A label absent from `space_dict` is a *non-materialising* leg (e.g.
            # the "r" label in the real-world fragmented example): it contributes
            # no tensor leg, so fragments sharing one SpaceID{N} can have tensor
            # rank < N. Existing scenarios list every label, so nothing is dropped.
            spaces = [sid.adjoint[i] ? space_dict[sid.labels[i]]' : space_dict[sid.labels[i]]
                      for i in 1:length(sid.labels) if haskey(space_dict, sid.labels[i])]
            isempty(spaces) && return ProductSpace{S_type, 0}()
            return ⊗(spaces...)
        end

        function build_random_A(keys_list, space_dict; is_hermitian=false, T=ComplexF64)
            S_type = spacetype(typeof(first(values(space_dict))))
            # Rank-agnostic value type: fragments may have different tensor ranks
            # (see `get_product_space` — non-materialising labels), so the dict
            # must not pin N₁/N₂ to the first key's rank.
            T_type = TensorMap{T, S_type, N₁, N₂, Vector{T}} where {N₁, N₂}
            data = Dictionary{Tuple{typeof(first(keys_list)[1]), typeof(first(keys_list)[2])}, T_type}()
            for (S_out, S_in) in keys_list
                haskey(data, (S_out, S_in)) && continue
                W_out = get_product_space(S_out, space_dict)
                W_in  = get_product_space(S_in,  space_dict)
                if is_hermitian && S_out == S_in
                    t = randn(T, W_out ← W_in)
                    insert!(data, (S_out, S_in), t + t')
                elseif is_hermitian && (S_in, S_out) in keys_list
                    t = randn(T, W_out ← W_in)
                    insert!(data, (S_out, S_in), t)
                    insert!(data, (S_in, S_out), copy(t'))
                else
                    insert!(data, (S_out, S_in), randn(T, W_out ← W_in))
                end
            end
            N_out = length(first(keys_list)[1].labels)
            N_in  = length(first(keys_list)[2].labels)
            return FragmentedTensor{N_out, N_in, T_type}(data)
        end

        # Build a fresh random perturbation matching A's key structure.
        # For hermitian scenarios, the perturbation is also hermitian (so that
        # `A + tΔ` stays hermitian for the eigh path to apply).
        function build_random_perturbation(A::FragmentedTensor{N_out, N_in, T}; hermitian=false) where {N_out, N_in, T}
            data = Dictionary{Tuple{SpaceID{N_out}, SpaceID{N_in}}, T}()
            for (k, v) in pairs(A.data)
                insert!(data, k, randn(eltype(v), space(v)))
            end
            res = FragmentedTensor{N_out, N_in, T}(data)
            return hermitian ? (res + res') / 2 : res
        end

        # `M` for the loss `real(dot(A_recon, M))` is non-hermitian random with
        # the same keyset (gauge-invariant losses don't require any hermicity
        # constraint on `M`).
        function build_random_M(A::FragmentedTensor{N_out, N_in, T}) where {N_out, N_in, T}
            data = Dictionary{Tuple{SpaceID{N_out}, SpaceID{N_in}}, T}()
            for (k, v) in pairs(A.data)
                insert!(data, k, randn(eltype(v), space(v)))
            end
            return FragmentedTensor{N_out, N_in, T}(data)
        end

        # Condition number of the eigenvector matrix of `eig_full(A)`.
        function eig_U_condition_number(A::FragmentedTensor; eigen_space_name="cond")
            D, U = eig_full(A; eigen_space_name)
            U_global, _ = FragmentedTensors.assemble_U_frag(U)
            # Sum of cond across blocksectors — for trivial sector it's just the one.
            T = scalartype(typeof(U_global))
            max_cond = zero(real(T))
            for c in blocksectors(U_global)
                cnum = cond(block(U_global, c))
                if cnum > max_cond
                    max_cond = cnum
                end
            end
            return max_cond
        end

        # AD comparison primitive: returns the (zyg, fd, rel_err) triple.
        function ad_compare(loss_fn)
            zyg = Zygote.gradient(loss_fn, 0.0)[1]
            fd  = FiniteDifferences.grad(central_fdm(5, 1), loss_fn, 0.0)[1]
            return zyg, fd
        end

        ad_ok(zyg, fd; atol=1e-5, rtol=1e-5) = isapprox(zyg, fd; atol, rtol)

        # ---- fuzz machinery ----
        #
        # A counter is a `(pass, excused_ill_cond, excused_near_boundary,
        # real_fail)` 4-tuple stored as a `MVector`-like vector for in-place
        # increments (Vector{Int} works fine).

        new_counter() = [0, 0, 0, 0]
        bump_pass!(c)         = (c[1] += 1; c)
        bump_ill_cond!(c)     = (c[2] += 1; c)
        bump_near_boundary!(c)= (c[3] += 1; c)
        bump_real_fail!(c)    = (c[4] += 1; c)

        # Smallest |kept eigenvalue / singular value| in a singleton
        # `FragmentedTensor{1,1}` (the D / S returned by truncated decomp).
        # Returns `Inf` if the FragmentedTensor is empty (no kept values).
        function min_abs_kept_value(D_kept::FragmentedTensor)
            isempty(D_kept.data) && return Inf
            diag_tm = first(values(D_kept.data))
            min_v = Inf
            for c in blocksectors(diag_tm)
                b = block(diag_tm, c)
                n = min(size(b, 1), size(b, 2))
                for i in 1:n
                    v = abs(b[i, i])
                    v < min_v && (min_v = v)
                end
            end
            return min_v
        end

        function run_fuzz(scn, n_reps;
                          ILL_COND_THRESHOLD = 1e6,
                          BOUNDARY_GAP_THRESHOLD = 1e-6,
                          TRUNC_ATOL = 1e-5,
                          atol = 1e-5,
                          rtol = 1e-5)
            counters = Dict{Symbol, Vector{Int}}(
                :eigh_full  => new_counter(),
                :eig_full   => new_counter(),
                :svd_full   => new_counter(),
                :eigh_trunc => new_counter(),
                :svd_trunc  => new_counter(),
            )
            recon_pass  = Dict{Symbol, Int}(:eigh_full => 0, :eig_full => 0, :svd_full => 0)
            recon_total = Dict{Symbol, Int}(:eigh_full => 0, :eig_full => 0, :svd_full => 0)

            for rep in 1:n_reps
                A = build_random_A(scn.keys, scn.space_dict; is_hermitian=scn.is_hermitian)
                N_out = length(first(keys(A.data))[1].labels)
                N_in  = length(first(keys(A.data))[2].labels)
                Δ = build_random_perturbation(A; hermitian=scn.is_hermitian)
                M = build_random_M(A)

                # --- eigh ---
                if scn.is_hermitian && N_out == N_in
                    D, U = eigh_full(A; eigen_space_name="eh")
                    recon_total[:eigh_full] += 1
                    if norm(U * D * U' - A) < 1e-9
                        recon_pass[:eigh_full] += 1
                    end

                    loss_eigh(t) = let A_t = A + t * Δ
                        Dt, Ut = eigh_full(A_t; eigen_space_name="eh")
                        real(dot(Ut * Dt * Ut', M))
                    end
                    zyg, fd = ad_compare(loss_eigh)
                    if ad_ok(zyg, fd; atol, rtol)
                        bump_pass!(counters[:eigh_full])
                    else
                        bump_real_fail!(counters[:eigh_full])
                    end
                end

                # --- eig ---
                if N_out == N_in
                    D, U = eig_full(A; eigen_space_name="eg")
                    recon_total[:eig_full] += 1
                    if norm(A * U - U * D) < 1e-9
                        recon_pass[:eig_full] += 1
                    end

                    loss_eig(t) = let A_t = A + t * Δ
                        Dt, Ut = eig_full(A_t; eigen_space_name="eg")
                        real(dot(Ut * Dt * inv(Ut), M))
                    end
                    zyg, fd = ad_compare(loss_eig)
                    if ad_ok(zyg, fd; atol, rtol)
                        bump_pass!(counters[:eig_full])
                    else
                        cond_U = eig_U_condition_number(A; eigen_space_name="eg")
                        if cond_U > ILL_COND_THRESHOLD
                            bump_ill_cond!(counters[:eig_full])
                        else
                            bump_real_fail!(counters[:eig_full])
                        end
                    end
                end

                # --- svd ---
                Us, Ss, Vs = svd_full(A; eigen_space_name="sv")
                recon_total[:svd_full] += 1
                if norm(Us * Ss * Vs - A) < 1e-9
                    recon_pass[:svd_full] += 1
                end

                loss_svd(t) = let A_t = A + t * Δ
                    Ut, St, Vt = svd_full(A_t; eigen_space_name="sv")
                    real(dot(Ut * St * Vt, M))
                end
                zyg, fd = ad_compare(loss_svd)
                if ad_ok(zyg, fd; atol, rtol)
                    bump_pass!(counters[:svd_full])
                else
                    bump_real_fail!(counters[:svd_full])
                end

                # --- eigh_trunc ---
                # Uses adjoint (no inv needed), so only the boundary-gap
                # excuse can apply (no ill-cond branch).
                if scn.is_hermitian && N_out == N_in
                    loss_eigh_trunc(t) = let A_t = A + t * Δ
                        Dk, Uk = eigh_trunc(A_t, trunctol(atol=TRUNC_ATOL); eigen_space_name="eht")
                        real(dot(Uk * Dk * Uk', M))
                    end
                    zyg_t, fd_t = ad_compare(loss_eigh_trunc)
                    if ad_ok(zyg_t, fd_t; atol, rtol)
                        bump_pass!(counters[:eigh_trunc])
                    else
                        Dk, _ = eigh_trunc(A, trunctol(atol=TRUNC_ATOL); eigen_space_name="eht")
                        gap = min_abs_kept_value(Dk) - TRUNC_ATOL
                        if gap < BOUNDARY_GAP_THRESHOLD
                            bump_near_boundary!(counters[:eigh_trunc])
                        else
                            bump_real_fail!(counters[:eigh_trunc])
                        end
                    end
                end

                # --- svd_trunc ---
                loss_svd_trunc(t) = let A_t = A + t * Δ
                    Uk, Sk, Vk = svd_trunc(A_t, trunctol(atol=TRUNC_ATOL); eigen_space_name="svt")
                    real(dot(Uk * Sk * Vk, M))
                end
                zyg_t, fd_t = ad_compare(loss_svd_trunc)
                if ad_ok(zyg_t, fd_t; atol, rtol)
                    bump_pass!(counters[:svd_trunc])
                else
                    _, Sk, _ = svd_trunc(A, trunctol(atol=TRUNC_ATOL); eigen_space_name="svt")
                    gap = min_abs_kept_value(Sk) - TRUNC_ATOL
                    if gap < BOUNDARY_GAP_THRESHOLD
                        bump_near_boundary!(counters[:svd_trunc])
                    else
                        bump_real_fail!(counters[:svd_trunc])
                    end
                end

                # --- eig_trunc — AD not tested ---
                # `eig_trunc` returns a non-square `U_kept`; reconstruction of
                # the kept part requires a pseudoinverse (or the left-eigenvector
                # matrix, which MAKit's API doesn't return). Until we add
                # `pinv` support for FragmentedTensor we can't exercise this
                # AD path with a gauge-invariant loss. See FACTORIZATIONS_TEST_PLAN.md.
            end

            # ---- aggregate assertions ----

            for decomp in (:eigh_full, :eig_full, :svd_full, :eigh_trunc, :svd_trunc)
                # Reconstruction tests are only run for full decompositions
                # (truncated decomps don't satisfy strict reconstruction).
                total = get(recon_total, decomp, 0)
                if total > 0
                    @test recon_pass[decomp] == total
                end

                c = counters[decomp]
                pass, ill, near, fail = c[1], c[2], c[3], c[4]
                total_ad = pass + ill + near + fail
                if total_ad > 0
                    @test fail == 0
                    @test pass > 0
                    @info "$(scn.name) | $decomp" pass excused_ill_cond=ill excused_near_boundary=near real_fail=fail
                end
            end
        end

        # ---- scenario catalogue ----

        N_REPS = 40
        SCENARIOS = NamedTuple[]

        # Group A: single SpaceID, hermitian, varying N.
        for N in (1, 2, 3)
            S = SpaceID{N}(NTuple{N,String}(fill("A", N)), NTuple{N,Bool}(fill(false, N)))
            push!(SCENARIOS, (
                name = "1 SpaceID N=$N hermitian",
                space_dict = Dict("A" => ℂ^2),
                keys = [(S, S)],
                is_hermitian = true,
            ))
        end

        # Group B: two SpaceIDs square, dim mix, key-pattern coverage.
        for (N, dimA, dimB) in [(1, 2, 3), (2, 2, 2), (2, 2, 3)]
            sd = Dict("A" => ℂ^dimA, "B" => ℂ^dimB)
            S1 = SpaceID{N}(NTuple{N,String}(fill("A", N)), NTuple{N,Bool}(fill(false, N)))
            S2 = SpaceID{N}(NTuple{N,String}(fill("B", N)), NTuple{N,Bool}(fill(false, N)))
            patterns = [
                ("dense 2x2 herm",       [(S1,S1),(S1,S2),(S2,S1),(S2,S2)], true),
                ("diagonal herm",        [(S1,S1),(S2,S2)],                 true),
                ("off-diag herm",        [(S1,S2),(S2,S1)],                 true),
                ("upper-tri full",       [(S1,S1),(S1,S2),(S2,S2)],         false),
                ("upper-tri partial",    [(S1,S1),(S1,S2)],                 false),
            ]
            for (desc, ks, herm) in patterns
                push!(SCENARIOS, (
                    name = "2 SpaceIDs N=$N dim=($dimA,$dimB) / $desc",
                    space_dict = sd,
                    keys = ks,
                    is_hermitian = herm,
                ))
            end
        end

        # Group C: three SpaceIDs, square N=2, mixed dims.
        # Note: strictly-triangular DAG patterns (nilpotent matrices) are
        # excluded per plan v3 §1.3 — TensorKit's eig returns a degenerate
        # decomposition and `inv(U)` doesn't exist.
        let
            sd = Dict("A" => ℂ^2, "B" => ℂ^2, "C" => ℂ^3)
            S1 = SpaceID{2}(("A","A"), (false,false))
            S2 = SpaceID{2}(("B","B"), (false,false))
            S3 = SpaceID{2}(("C","C"), (false,false))
            patterns = [
                ("3x3 dense herm",
                    [(Si,Sj) for Si in (S1,S2,S3) for Sj in (S1,S2,S3)],
                    true),
                ("3x3 diagonal herm",
                    [(S1,S1),(S2,S2),(S3,S3)],
                    true),
                ("3x3 off-diag herm",
                    [(S1,S2),(S1,S3),(S2,S1),(S2,S3),(S3,S1),(S3,S2)],
                    true),
                ("3x3 upper-tri full",
                    [(S1,S1),(S1,S2),(S1,S3),(S2,S2),(S2,S3),(S3,S3)],
                    false),
            ]
            for (desc, ks, herm) in patterns
                push!(SCENARIOS, (
                    name = "3 SpaceIDs N=2 / $desc",
                    space_dict = sd,
                    keys = ks,
                    is_hermitian = herm,
                ))
            end
        end

        # Group D: rectangular cases (svd only).
        let
            sd = Dict("A" => ℂ^2, "B" => ℂ^3, "C" => ℂ^2)
            for (No, Ni, lout, lin) in [
                    (1, 2, "A", "B"),
                    (2, 1, "A", "B"),
                    (2, 3, "A", "B"),
                ]
                SO = SpaceID{No}(NTuple{No,String}(fill(lout, No)), NTuple{No,Bool}(fill(false, No)))
                SI = SpaceID{Ni}(NTuple{Ni,String}(fill(lin, Ni)), NTuple{Ni,Bool}(fill(false, Ni)))
                push!(SCENARIOS, (
                    name = "rectangular N_out=$No N_in=$Ni single-key",
                    space_dict = sd,
                    keys = [(SO, SI)],
                    is_hermitian = false,
                ))
            end
        end

        # Group E: graded space (Z2).
        let
            V_z2 = Z2Space(0 => 2, 1 => 1)
            sd = Dict("A" => V_z2, "B" => V_z2)
            S1 = SpaceID{1}(("A",), (false,))
            S2 = SpaceID{1}(("B",), (false,))
            patterns = [
                ("Z2 dense herm",    [(S1,S1),(S1,S2),(S2,S1),(S2,S2)], true),
                ("Z2 diagonal herm", [(S1,S1),(S2,S2)],                 true),
                ("Z2 upper-tri",     [(S1,S1),(S1,S2),(S2,S2)],         false),
            ]
            for (desc, ks, herm) in patterns
                push!(SCENARIOS, (
                    name = desc,
                    space_dict = sd,
                    keys = ks,
                    is_hermitian = herm,
                ))
            end
        end

        # Group F: fragmented *varying-rank* fragments. A SpaceID{5} whose "r"
        # legs do not materialise (absent from `space_dict`), so fragments under
        # one N=5 key type carry tensor ranks 5 / 3 / 1. This mirrors the
        # real-world failing example (`FragTens/failing_example.data`) at reduced
        # dims (its Z2(8,8) → Z2(3,3); nv legs minimal). Guards every rank-pinning
        # site: assemble/disassemble (forward) AND the `dot` pullback (AD).
        # Both hermitian (eigh path) and non-hermitian (eig/svd path) are covered.
        # Fewer reps: the rank-5 global block makes the AD heavier than the
        # ℂ-space scenarios, and the structure — not the RNG — is what matters.
        let
            sd = Dict("nv" => Z2Space(0 => 1, 1 => 1), "nh" => Z2Space(0 => 3, 1 => 3))
            adj = (true, false, false, false, true)
            S_full = SpaceID{5}(("nv", "nv", "nh", "nv", "nv"), adj)  # rank 5
            S_lo   = SpaceID{5}(("nv", "nv", "nh", "r",  "r"),  adj)  # rank 3
            S_ro   = SpaceID{5}(("r",  "r",  "nh", "nv", "nv"), adj)  # rank 3
            S_min  = SpaceID{5}(("r",  "r",  "nh", "r",  "r"),  adj)  # rank 1
            sp = [S_full, S_lo, S_ro, S_min]
            dense = [(Si, Sj) for Si in sp for Sj in sp]
            push!(SCENARIOS, (
                name = "fragmented varying-rank N=5 dense herm",
                space_dict = sd, keys = dense, is_hermitian = true, reps = 6,
            ))
            push!(SCENARIOS, (
                name = "fragmented varying-rank N=5 dense non-herm",
                space_dict = sd, keys = dense, is_hermitian = false, reps = 6,
            ))
        end

        # ---- run all scenarios ----

        @testset "[$(lpad(i, 2))] $(scn.name)" for (i, scn) in enumerate(SCENARIOS)
            Random.seed!(0xC0DE_FACE + i * 0x9E37_79B9)
            run_fuzz(scn, hasproperty(scn, :reps) ? scn.reps : N_REPS)
        end

        # ---- edge case: empty FragmentedTensor ----
        @testset "Empty tensor: norm = 0" begin
            for (N_out, N_in) in [(1,1), (2,2), (1,2), (2,3)]
                T_type = TensorMap{ComplexF64, ComplexSpace, N_out, N_in, Vector{ComplexF64}}
                A_empty = FragmentedTensor{N_out, N_in, T_type}()
                @test isempty(A_empty.data)
                @test norm(A_empty) == 0.0
            end
        end
    end

    @testset "Fragmented Multi-Space and Multi-Leg Factorizations" begin
        # Setup Spaces
        V = ℂ^2
        S_A = SpaceID{1}(("A",), (false,))
        S_B = SpaceID{1}(("B",), (false,))
        S_C = SpaceID{1}(("C",), (false,))
        
        # Helper to construct a random TensorMap
        rand_tm(codom, dom) = randn(ComplexF64, codom ← dom)
        
        # 1. 2 Spaces, All-to-all entries (Hermitian)
        H_AA = rand_tm(V, V); H_AA = H_AA + H_AA'
        H_BB = rand_tm(V, V); H_BB = H_BB + H_BB'
        H_AB = rand_tm(V, V)
        H_BA = H_AB'
        
        A_2all = FragmentedTensor(Dictionary(
            [(S_A, S_A), (S_A, S_B), (S_B, S_A), (S_B, S_B)],
            [H_AA, H_AB, H_BA, H_BB]
        ))
        
        D, U = eigh_full(A_2all; eigen_space_name="eigen_2all")
        A_recon = U * D * U'
        @test norm(A_recon - A_2all) < 1e-12
        
        # 2 Spaces, All-to-all (General eig and SVD)
        G_AA = rand_tm(V, V); G_BB = rand_tm(V, V)
        G_AB = rand_tm(V, V); G_BA = rand_tm(V, V)
        A_2gen = FragmentedTensor(Dictionary(
            [(S_A, S_A), (S_A, S_B), (S_B, S_A), (S_B, S_B)],
            [G_AA, G_AB, G_BA, G_BB]
        ))
        
        D_g, U_g = eig_full(A_2gen; eigen_space_name="eigen_2gen")
        # Check defining relation: A * U ≈ U * D
        @test norm(A_2gen * U_g - U_g * D_g) < 1e-12
        
        U_s, S_s, V_s = svd_full(A_2gen; eigen_space_name="svd_2gen")
        @test norm(U_s * S_s * V_s - A_2gen) < 1e-12
        
        U_sc, S_sc, V_sc = svd_compact(A_2gen; eigen_space_name="svd_2gen_c")
        @test norm(U_sc * S_sc * V_sc - A_2gen) < 1e-12
        
        # 2. 2 Spaces, Partial Connectivity (Non-Hermitian)
        A_2part = FragmentedTensor(Dictionary(
            [(S_A, S_A), (S_A, S_B), (S_B, S_B)],
            [G_AA, G_AB, G_BB]
        ))
        
        D_p, U_p = eig_full(A_2part; eigen_space_name="eigen_2part")
        @test norm(A_2part * U_p - U_p * D_p) < 1e-12
        
        U_sp, S_sp, V_sp = svd_full(A_2part; eigen_space_name="svd_2part")
        @test norm(U_sp * S_sp * V_sp - A_2part) < 1e-12
        
        # 3. 3 Spaces, All-to-all
        H_CC = rand_tm(V, V); H_CC = H_CC + H_CC'
        H_AC = rand_tm(V, V); H_CA = H_AC'
        H_BC = rand_tm(V, V); H_CB = H_BC'
        
        A_3all = FragmentedTensor(Dictionary(
            [(S_A, S_A), (S_A, S_B), (S_A, S_C), 
             (S_B, S_A), (S_B, S_B), (S_B, S_C),
             (S_C, S_A), (S_C, S_B), (S_C, S_C)],
            [H_AA, H_AB, H_AC, H_BA, H_BB, H_BC, H_CA, H_CB, H_CC]
        ))
        
        D_3, U_3 = eigh_full(A_3all; eigen_space_name="eigen_3all")
        @test norm(U_3 * D_3 * U_3' - A_3all) < 1e-12
        
        # 4. 3 Spaces, Partial Connectivity
        A_3part = FragmentedTensor(Dictionary(
            [(S_A, S_A), (S_A, S_B), (S_B, S_C), (S_C, S_A)],
            [G_AA, G_AB, rand_tm(V, V), rand_tm(V, V)]
        ))
        
        D_3p, U_3p = eig_full(A_3part; eigen_space_name="eigen_3part")
        @test norm(A_3part * U_3p - U_3p * D_3p) < 1e-12
        
        U_3sp, S_3sp, V_3sp = svd_full(A_3part; eigen_space_name="svd_3part")
        @test norm(U_3sp * S_3sp * V_3sp - A_3part) < 1e-12
        
        # 5. 3 Spaces, Multi-Leg: spaces 1,2 in codomain, 2,3 in domain
        S_out1 = SpaceID{2}(("1", "2"), (false, false))
        S_out2 = SpaceID{2}(("2", "4"), (false, false))
        S_in1  = SpaceID{2}(("2", "3"), (false, false))
        S_in2  = SpaceID{2}(("3", "5"), (false, false))
        
        t_ml11 = rand_tm(V ⊗ V, V ⊗ V)
        t_ml12 = rand_tm(V ⊗ V, V ⊗ V)
        t_ml21 = rand_tm(V ⊗ V, V ⊗ V)
        t_ml22 = rand_tm(V ⊗ V, V ⊗ V)
        
        A_ml = FragmentedTensor(Dictionary(
            [(S_out1, S_in1), (S_out1, S_in2), (S_out2, S_in1), (S_out2, S_in2)],
            [t_ml11, t_ml12, t_ml21, t_ml22]
        ))
        
        U_ml, S_ml, V_ml = svd_full(A_ml; eigen_space_name="svd_ml")
        @test norm(U_ml * S_ml * V_ml - A_ml) < 1e-12
        
        # 6. 4 Spaces, 1,2 in domain, 3,4 in codomain
        S_out_4a = SpaceID{2}(("3", "4"), (false, false))
        S_in_4a  = SpaceID{2}(("1", "2"), (false, false))
        
        A_ml4a = FragmentedTensor(Dictionary(
            [(S_out_4a, S_in_4a)],
            [t_ml11]
        ))
        
        U_4a, S_4a, V_4a = svd_full(A_ml4a; eigen_space_name="svd_4a")
        @test norm(U_4a * S_4a * V_4a - A_ml4a) < 1e-12
        
        # 7. 4 Spaces, 1,2,3 in domain, 4 in codomain
        S_out_4b = SpaceID{1}(("4",), (false,))
        S_in_4b  = SpaceID{3}(("1", "2", "3"), (false, false, false))
        
        t_ml3 = rand_tm(V, V ⊗ V ⊗ V)
        A_ml4b = FragmentedTensor(Dictionary(
            [(S_out_4b, S_in_4b)],
            [t_ml3]
        ))
        
        U_4b, S_4b, V_4b = svd_full(A_ml4b; eigen_space_name="svd_4b")
        @test norm(U_4b * S_4b * V_4b - A_ml4b) < 1e-12

        # 8. Fragmented VARYING-RANK fragments. A SpaceID{5} whose "r" legs do
        #    not materialise as tensor legs, so fragments sharing one N=5 key
        #    type carry tensor ranks 5 / 3 / 1. This is the structure of the
        #    real-world failing example (`failing_example.data`) at reduced dims
        #    (its Z2(8,8) → Z2(3,3)). It used to crash every decomposition with a
        #    `convert` MethodError because the assemble/disassemble containers
        #    pinned the ProductSpace / TensorMap rank to N. Explicit reconstruction
        #    guard (no AD) for both hermitian and non-hermitian tensors.
        let
            sdv = Dict("nv" => Z2Space(0 => 1, 1 => 1), "nh" => Z2Space(0 => 3, 1 => 3))
            S_type = typeof(sdv["nh"])
            getps(sid) = ⊗([sid.adjoint[i] ? sdv[sid.labels[i]]' : sdv[sid.labels[i]]
                            for i in 1:length(sid.labels) if haskey(sdv, sid.labels[i])]...)
            adjp = (true, false, false, false, true)
            S_full = SpaceID{5}(("nv", "nv", "nh", "nv", "nv"), adjp)
            S_lo   = SpaceID{5}(("nv", "nv", "nh", "r",  "r"),  adjp)
            S_ro   = SpaceID{5}(("r",  "r",  "nh", "nv", "nv"), adjp)
            S_min  = SpaceID{5}(("r",  "r",  "nh", "r",  "r"),  adjp)
            fsp = [S_full, S_lo, S_ro, S_min]
            Tval = TensorMap{ComplexF64, S_type, Na, Nb, Vector{ComplexF64}} where {Na, Nb}

            # The defining feature: same SpaceID{5} N, but tensor ranks differ.
            @test length(getps(S_full)) == 5
            @test length(getps(S_lo)) == 3
            @test length(getps(S_min)) == 1

            # Hermitian fragmented tensor → eigh.
            dh = Dictionary{Tuple{SpaceID{5}, SpaceID{5}}, Tval}()
            for So in fsp, Si in fsp
                haskey(dh, (So, Si)) && continue
                Wo, Wi = getps(So), getps(Si)
                if So == Si
                    t = randn(ComplexF64, Wo ← Wi)
                    set!(dh, (So, Si), t + t')
                else
                    t = randn(ComplexF64, Wo ← Wi)
                    set!(dh, (So, Si), t)
                    set!(dh, (Si, So), copy(t'))
                end
            end
            A_fragh = FragmentedTensor{5, 5, Tval}(dh)
            Dh, Uh = eigh_full(A_fragh; eigen_space_name="frag_eigh")
            @test norm(Uh * Dh * Uh' - A_fragh) < 1e-10

            # Non-hermitian fragmented tensor → eig + svd.
            dn = Dictionary{Tuple{SpaceID{5}, SpaceID{5}}, Tval}()
            for So in fsp, Si in fsp
                set!(dn, (So, Si), randn(ComplexF64, getps(So) ← getps(Si)))
            end
            A_fragn = FragmentedTensor{5, 5, Tval}(dn)
            Dg, Ug = eig_full(A_fragn; eigen_space_name="frag_eig")
            @test norm(A_fragn * Ug - Ug * Dg) < 1e-10
            @test norm(Ug * Dg * inv(Ug) - A_fragn) < 1e-10
            Usf, Ssf, Vsf = svd_full(A_fragn; eigen_space_name="frag_svd")
            @test norm(Usf * Ssf * Vsf - A_fragn) < 1e-10
        end
    end
end
