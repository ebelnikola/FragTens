using Test
using Dictionaries
using LinearAlgebra
using VectorInterface
using TensorKit
using FragmentedTensors
using FragmentedTensors: SpaceID, FragmentedTensor, NullSpaceID

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
        # Key sets different -> zero
        @test dot(FT1, FT2) == 0.0
        
        # Key sets identical -> sum key-by-key
        FT3 = FragmentedTensor(Dictionary([(sid"A", sid"B"), (sid"C", sid"D")], [3.0, 4.0]))
        @test dot(FT1, FT3) ≈ 1.0*3.0 + 2.0*4.0
        
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
    @testset "Inhomogeneous / Heterogeneous Tensors" begin
        # 1. Using Matrix of differing sizes (heterogeneous dimensions)
        m_2x2 = [1.0 2.0; 3.0 4.0]
        m_3x3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        
        # We specify the element type as Any to allow varying Matrix sizes
        FT_A = FragmentedTensor{1, 1, Any}(Dictionary([(sid"i", sid"j"), (sid"x", sid"y")], Any[m_2x2, m_3x3]))
        FT_B = FragmentedTensor{1, 1, Any}(Dictionary([(sid"j", sid"k"), (sid"y", sid"z")], Any[m_2x2, m_3x3]))
        
        FT_C = FT_A * FT_B
        @test FT_C[(sid"i", sid"k")] ≈ m_2x2 * m_2x2
        @test FT_C[(sid"x", sid"z")] ≈ m_3x3 * m_3x3
        @test valtype(FT_C.data) === Any

        # 2. Using TensorMap with differing leg counts (heterogeneous leg counts)
        # We define a FragmentedTensor with Any to hold maps with 1 out-1 in and 2 out-2 in
        V = ℂ^2
        t_1_1 = randn(ComplexF64, V, V)
        t_2_2 = randn(ComplexF64, V ⊗ V, V ⊗ V)
        
        FT_A_tm = FragmentedTensor{1, 1, Any}(Dictionary([(sid"a", sid"b"), (sid"x", sid"y")], Any[t_1_1, t_2_2]))
        FT_B_tm = FragmentedTensor{1, 1, Any}(Dictionary([(sid"b", sid"c"), (sid"y", sid"z")], Any[t_1_1, t_2_2]))
        
        FT_C_tm = FT_A_tm * FT_B_tm
        @test FT_C_tm[(sid"a", sid"c")] ≈ t_1_1 * t_1_1
        @test FT_C_tm[(sid"x", sid"z")] ≈ t_2_2 * t_2_2
        @test valtype(FT_C_tm.data) === Any
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
end
