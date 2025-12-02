import streamlit as st
import pandas as pd
import pulp

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(layout="wide", page_title="Tối Ưu Hóa Bữa Trưa (Linear Programming)")

def run_optimization(foods_data: dict):
    """
    Hàm giải mô hình tối ưu hóa ăn trưa sử dụng PuLP.
    Trả về kết quả (cost, solution) hoặc None nếu không khả thi.
    """
    if not foods_data:
        return None, "Lỗi: Không có dữ liệu thực phẩm để chạy mô hình."

    food_names = list(foods_data.keys())
    model = pulp.LpProblem("Lunch Optimization Flexible", pulp.LpMinimize)
    
    # Biến quyết định (số nguyên không âm)
    # Tên biến PuLP phải là chuỗi không chứa ký tự đặc biệt, nên dùng food_name trực tiếp.
    x = pulp.LpVariable.dicts("X", food_names, lowBound=0, cat='Integer')

    # --- 1. HÀM MỤC TIÊU (Minimize Cost) ---
    model += (
        pulp.lpSum(foods_data[name]['cost'] * x[name] for name in food_names),
        "Total_Cost"
    )

    # Tính toán biểu thức tổng Calo và Calo từ béo
    Total_Cal_Expr = pulp.lpSum(foods_data[name]['total_cal'] * x[name] for name in food_names)
    Cal_Fat_Expr = pulp.lpSum(foods_data[name]['cal_fat'] * x[name] for name in food_names)

    # --- 2. RÀNG BUỘC (Constraints) ---
    # C1, C2: Total Calories (Min 400, Max 600)
    model += (Total_Cal_Expr >= 400, "Min_Total_Calories")
    model += (Total_Cal_Expr <= 600, "Max_Total_Calories")

    # C3: Max 30% Calories from Fat
    model += (Cal_Fat_Expr - 0.30 * Total_Cal_Expr <= 0, "Max_30_Percent_Fat_Calories")

    # C4: Vitamin C (Min 60 mg)
    model += (pulp.lpSum(foods_data[name]['vit_c'] * x[name] for name in food_names) >= 60, "Min_Vitamin_C")

    # C5: Protein (Min 12 g)
    model += (pulp.lpSum(foods_data[name]['protein'] * x[name] for name in food_names) >= 12, "Min_Protein")
    
    # --- RÀNG BUỘC ĐẶC BIỆT (Kiểm tra tồn tại) ---
    if 'bread' in food_names:
        model += (x['bread'] == 2, "Exact_2_Slices_Bread")

    if 'peanut_butter' in food_names and 'jelly' in food_names:
        model += (x['peanut_butter'] - 2 * x['jelly'] >= 0, "Peanut_Butter_vs_Jelly")

    liquid_items = [name for name in ['milk', 'juice'] if name in food_names]
    if liquid_items:
        model += (pulp.lpSum(x[name] for name in liquid_items) >= 1, "Min_1_Cup_Liquid")
    
    # --- 3. GIẢI MÔ HÌNH ---
    model.solve()

    if model.status == pulp.LpStatusOptimal:
        optimal_cost = pulp.value(model.objective)
        results = {name: int(round(x[name].varValue)) for name in food_names}
        return optimal_cost, results
    
    return None, pulp.LpStatus[model.status]

# --- DỮ LIỆU MẶC ĐỊNH CHO BẢNG ---
DEFAULT_FOODS = {
    'bread': {'cost': 5, 'cal_fat': 10, 'total_cal': 70, 'vit_c': 0, 'protein': 3},
    'peanut_butter': {'cost': 4, 'cal_fat': 75, 'total_cal': 100, 'vit_c': 0, 'protein': 4},
    'jelly': {'cost': 7, 'cal_fat': 0, 'total_cal': 50, 'vit_c': 3, 'protein': 0},
    'cracker': {'cost': 8, 'cal_fat': 20, 'total_cal': 60, 'vit_c': 0, 'protein': 1},
    'milk': {'cost': 15, 'cal_fat': 70, 'total_cal': 150, 'vit_c': 2, 'protein': 8},
    'juice': {'cost': 35, 'cal_fat': 0, 'total_cal': 100, 'vit_c': 120, 'protein': 1}
}

# Chuyển đổi từ dict sang DataFrame cho Streamlit
df_default = pd.DataFrame.from_dict(DEFAULT_FOODS, orient='index')
df_default.index.name = 'food_name'


# --- HÀM CHÍNH CỦA STREAMLIT ---
def main():
    st.title("🥪 Tối Ưu Hóa Chi Phí Bữa Trưa")
    st.markdown("Sử dụng **Lập trình Tuyến tính** (PuLP) để tìm bữa ăn với chi phí thấp nhất đáp ứng yêu cầu dinh dưỡng.")
    
    st.divider()

    ## 1. PHẦN NHẬP DỮ LIỆU (Bảng tương tác)
    st.header("1. Nhập và Chỉnh Sửa Dữ Liệu Thực Phẩm")
    st.markdown("⚠️ **Lưu ý:** Tên món ăn cần viết liền không dấu, không khoảng trắng (ví dụ: `peanut_butter`).")
    st.markdown("Tất cả giá trị phải là số và $\ge 0$.")
    
    # Tạo bảng dữ liệu có thể chỉnh sửa
    edited_df = st.data_editor(
        df_default,
        # Đảm bảo các cột là số (float hoặc int)
        column_config={
            col: st.column_config.NumberColumn(
                f"{col.replace('_', ' ').title()}", 
                min_value=0.0,
                format="%.2f"
            ) for col in df_default.columns
        },
        num_rows="dynamic", # Cho phép thêm/xóa hàng
        use_container_width=True
    )
    
    # Chuyển DataFrame đã chỉnh sửa về dict cho PuLP
    foods_input = edited_df.to_dict('index')

    # --- KIỂM TRA LOGIC DỮ LIỆU (Total_Cal >= Cal_Fat) ---
    data_is_valid = True
    for food_name, data in foods_input.items():
        if data['cal_fat'] > data['total_cal']:
            st.error(
                f"❌ LỖI LOGIC: Món **{food_name}** có lượng Calories từ Chất Béo ({data['cal_fat']:.2f}) "
                f"lớn hơn Tổng Calo ({data['total_cal']:.2f}). Vui lòng sửa lại dữ liệu trong bảng."
            )
            data_is_valid = False
            break

    st.divider()

    ## 2. PHẦN CHẠY MÔ HÌNH VÀ KẾT QUẢ
    st.header("2. Kết quả tối ưu hóa")

    if st.button("Chạy mô hình tối ưu", disabled=not data_is_valid):
        
        # Chạy mô hình PuLP
        optimal_cost, result_data = run_optimization(foods_input)

        if optimal_cost is not None:
            st.success("✅ **ĐÃ TÌM THẤY KẾT QUẢ TỐI ƯU**")
            
            col1, col2 = st.columns(2)
            
            # Hiển thị Chi phí
            with col1:
                st.metric("Chi phí tối thiểu", f"{optimal_cost:.2f} ¢")
            
            # Tạo bảng kết quả số lượng
            solution_df = pd.DataFrame(
                result_data.items(), 
                columns=['Thực phẩm', 'Số lượng tối ưu']
            )
            solution_df['Số lượng tối ưu'] = solution_df['Số lượng tối ưu'].astype(int)
            
            with col2:
                 st.dataframe(solution_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")

            # --- KIỂM TRA CÁC RÀNG BUỘC SAU KHI TỐI ƯU ---
            st.subheader("Kiểm tra Dinh dưỡng Cuối cùng:")
            
            # Tính toán các thông số dinh dưỡng của giải pháp tối ưu
            total_cal = sum(foods_input[name]['total_cal'] * result_data[name] for name in result_data)
            cal_fat = sum(foods_input[name]['cal_fat'] * result_data[name] for name in result_data)
            vit_c = sum(foods_input[name]['vit_c'] * result_data[name] for name in result_data)
            protein = sum(foods_input[name]['protein'] * result_data[name] for name in result_data)
            
            st.table(pd.DataFrame({
                'Chỉ Số': ['Tổng Calo (kcal)', 'Calo từ chất béo (kcal)', 'Vitamin C (mg)', 'Protein (g)'],
                'Giá Trị Đạt Được': [f"{total_cal:.2f}", f"{cal_fat:.2f}", f"{vit_c:.2f}", f"{protein:.2f}"],
                'Yêu Cầu Ràng Buộc': [
                    '400 - 600', 
                    f'<= 30% ({0.3 * total_cal:.2f})', 
                    '>= 60', 
                    '>= 12'
                ]
            }))
            
        else:
            st.error(f"❌ KHÔNG TÌM THẤY LỜI GIẢI TỐI ƯU. **Trạng thái**: {result_data}. Vui lòng kiểm tra lại các ràng buộc hoặc dữ liệu nhập.")
    
    st.caption("Mô hình được giải quyết bằng PuLP (Integer Linear Programming).")

if __name__ == "__main__":

    main()
