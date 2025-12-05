import streamlit as st
import pandas as pd
import pulp
from typing import List, Dict, Any

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(layout="wide", page_title="Tối Ưu Hóa Bữa Trưa (Linear Programming)")

# Định nghĩa kiểu dữ liệu cho ràng buộc tùy chỉnh
CustomConstraint = Dict[str, Any]

def run_optimization(foods_data: dict, custom_constraints: List[CustomConstraint]):
    """
    Hàm giải mô hình tối ưu hóa ăn trưa sử dụng PuLP, có thêm ràng buộc tùy chỉnh.
    Trả về kết quả (cost, solution) hoặc None nếu không khả thi.
    """
    if not foods_data:
        return None, "Lỗi: Không có dữ liệu thực phẩm để chạy mô hình."

    food_names = list(foods_data.keys())
    model = pulp.LpProblem("Lunch Optimization Flexible", pulp.LpMinimize)
    
    # Biến quyết định (số nguyên không âm)
    x = pulp.LpVariable.dicts("X", food_names, lowBound=0, cat='Integer')

    # --- 1. HÀM MỤC TIÊU (Minimize Cost) ---
    model += (
        pulp.lpSum(foods_data[name]['cost'] * x[name] for name in food_names),
        "Total_Cost"
    )

    # Tính toán biểu thức tổng Calo và Calo từ béo
    Total_Cal_Expr = pulp.lpSum(foods_data[name]['total_cal'] * x[name] for name in food_names)
    Cal_Fat_Expr = pulp.lpSum(foods_data[name]['cal_fat'] * x[name] for name in food_names)

    # --- 2. RÀNG BUỘC CỐ ĐỊNH (Constraints) ---
    
    # C1, C2: Total Calories (Min 400, Max 600)
    model += (Total_Cal_Expr >= 400, "Fixed_Min_Total_Calories")
    model += (Total_Cal_Expr <= 600, "Fixed_Max_Total_Calories")

    # C3: Max 30% Calories from Fat
    model += (Cal_Fat_Expr - 0.30 * Total_Cal_Expr <= 0, "Fixed_Max_30_Percent_Fat_Calories")

    # C4: Vitamin C (Min 60 mg)
    model += (pulp.lpSum(foods_data[name]['vit_c'] * x[name] for name in food_names) >= 60, "Fixed_Min_Vitamin_C")

    # C5: Protein (Min 12 g)
    model += (pulp.lpSum(foods_data[name]['protein'] * x[name] for name in food_names) >= 12, "Fixed_Min_Protein")
    
    # RÀNG BUỘC ĐẶC BIỆT (Kiểm tra tồn tại)
    if 'bread' in food_names:
        model += (x['bread'] == 2, "Fixed_Exact_2_Slices_Bread")

    if 'peanut_butter' in food_names and 'jelly' in food_names:
        # Bơ đậu phộng >= 2 * Thạch (Để đảm bảo sandwich có đủ bơ)
        model += (x['peanut_butter'] - 2 * x['jelly'] >= 0, "Fixed_Peanut_Butter_vs_Jelly")

    liquid_items = [name for name in ['milk', 'juice'] if name in food_names]
    if liquid_items:
        model += (pulp.lpSum(x[name] for name in liquid_items) >= 1, "Fixed_Min_1_Cup_Liquid")

    # --- 3. RÀNG BUỘC TÙY CHỈNH (Custom Constraints) ---
    
    valid_food_attributes = foods_data[food_names[0]].keys() if food_names else []

    for i, constraint in enumerate(custom_constraints):
        nutrient = constraint.get('Nutrient', '').strip()
        operator = constraint.get('Operator', '').strip()
        value = constraint.get('Value', 0)

        # Bỏ qua các ràng buộc không hợp lệ
        if not nutrient or operator not in ['>=', '<=', '='] or nutrient not in valid_food_attributes:
            continue

        # Xây dựng biểu thức tuyến tính tổng (ví dụ: Tổng Protein)
        try:
            total_expr = pulp.lpSum(foods_data[name][nutrient] * x[name] for name in food_names)
            
            # Thêm ràng buộc vào mô hình
            if operator == '>=':
                model += (total_expr >= value, f"Custom_Constraint_{i+1}_{nutrient}_{operator}_{value}")
            elif operator == '<=':
                model += (total_expr <= value, f"Custom_Constraint_{i+1}_{nutrient}_{operator}_{value}")
            elif operator == '=':
                model += (total_expr == value, f"Custom_Constraint_{i+1}_{nutrient}_{operator}_{value}")
        except KeyError:
            # Điều này sẽ xảy ra nếu một cột bị thiếu trong dữ liệu đầu vào, 
            # nhưng đã được kiểm tra cơ bản bằng `nutrient not in valid_food_attributes`
            pass

    # --- 4. GIẢI MÔ HÌNH ---
    model.solve()

    if model.status == pulp.LpStatusOptimal:
        optimal_cost = pulp.value(model.objective)
        # Chỉ hiển thị những món có số lượng > 0
        results = {name: int(round(x[name].varValue)) 
                   for name in food_names 
                   if x[name].varValue is not None and x[name].varValue > 1e-6}
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

df_default = pd.DataFrame.from_dict(DEFAULT_FOODS, orient='index')
df_default.index.name = 'food_name'

# Cấu hình mặc định cho bảng ràng buộc
DEFAULT_CONSTRAINTS_DF = pd.DataFrame({
    'Nutrient': ['cost'],
    'Operator': ['<='],
    'Value': [200.0]
})


# --- HÀM CHÍNH CỦA STREAMLIT ---
def main():
    st.title("🥪 TỐI ƯU HÓA CHI PHÍ BỮA TRƯA")
    st.markdown("Sử dụng **Lập trình Tuyến tính Số nguyên** (PuLP) để tìm bữa ăn với chi phí thấp nhất đáp ứng yêu cầu dinh dưỡng và ràng buộc tùy chỉnh.")
    
    st.divider()

    ## 1. PHẦN NHẬP DỮ LIỆU THỰC PHẨM
    st.header("1. Nhập và chỉnh sửa dữ liệu thực phẩm")
    st.markdown("⚠️ **Lưu ý:** Tên món ăn nên viết liền không dấu, không khoảng trắng (ví dụ: `peanut_butter`).")
    st.markdown("Các cột là: **cost** (chi phí), **cal_fat** (calo từ béo), **total_cal** (tổng calo), **vit_c**, **protein**.")
    
    # Tạo bảng dữ liệu có thể chỉnh sửa
    edited_df = st.data_editor(
        df_default,
        column_config={
            col: st.column_config.NumberColumn(
                f"{col.replace('_', ' ').title()}", 
                min_value=0.0,
                format="%.2f"
            ) for col in df_default.columns
        },
        num_rows="dynamic",
        use_container_width=True,
        key="food_data_editor"
    )
    
    foods_input = edited_df.to_dict('index')

    # --- KIỂM TRA LOGIC DỮ LIỆU ---
    data_is_valid = True
    for food_name, data in foods_input.items():
        if data.get('cal_fat', 0) > data.get('total_cal', 0):
            st.error(
                f"❌ LỖI LOGIC: Món **{food_name}** có Calo từ béo ({data['cal_fat']:.2f}) "
                f"lớn hơn Tổng Calo ({data['total_cal']:.2f})."
            )
            data_is_valid = False
            break

    st.divider()
    
    ## 2. RÀNG BUỘC TÙY CHỈNH
    st.header("2. Thêm Ràng Buộc Tùy Chỉnh")
    st.markdown("Nhập các ràng buộc bổ sung theo cú pháp: **Tổng [Chất dinh dưỡng] [Toán tử] [Giá trị]**.")
    st.markdown("Ví dụ: `total_cal` $>= 500$, `protein` $<= 30$, `cost` $= 100$.")
    
    # Tên cột hợp lệ dựa trên dữ liệu thực phẩm
    valid_attributes = list(df_default.columns)
    operator_options = ['>=', '<=', '=']

    custom_constraints_df = st.data_editor(
        DEFAULT_CONSTRAINTS_DF,
        column_config={
            "Nutrient": st.column_config.SelectboxColumn(
                "Chất dinh dưỡng",
                options=valid_attributes,
                required=True,
                help="Chọn thuộc tính của thực phẩm (Tên cột)."
            ),
            "Operator": st.column_config.SelectboxColumn(
                "Toán tử",
                options=operator_options,
                required=True,
                help="Chọn toán tử so sánh (>=, <=, =)."
            ),
            "Value": st.column_config.NumberColumn(
                "Giá trị",
                min_value=0.0,
                format="%.2f",
                required=True,
                help="Nhập giá trị mục tiêu của ràng buộc."
            )
        },
        num_rows="dynamic",
        use_container_width=True,
        key="custom_constraints_editor"
    )
    
    custom_constraints = custom_constraints_df.to_dict('records')

    st.divider()

    ## 3. PHẦN CHẠY MÔ HÌNH VÀ KẾT QUẢ
    st.header("3. Kết quả tối ưu hóa")

    if st.button("🚀 Chạy mô hình tối ưu", disabled=not data_is_valid):
        
        # Chạy mô hình PuLP
        optimal_cost, result_data = run_optimization(foods_input, custom_constraints)

        if optimal_cost is not None:
            st.success("✅ **ĐÃ TÌM THẤY KẾT QUẢ TỐI ƯU**")
            
            col1, col2 = st.columns([1, 2])
            
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
            st.subheader("Kiểm tra Dinh dưỡng và Ràng buộc:")
            
            # Tính toán các thông số dinh dưỡng của giải pháp tối ưu
            total_cal = sum(foods_input[name]['total_cal'] * result_data.get(name, 0) for name in foods_input)
            cal_fat = sum(foods_input[name]['cal_fat'] * result_data.get(name, 0) for name in foods_input)
            vit_c = sum(foods_input[name]['vit_c'] * result_data.get(name, 0) for name in foods_input)
            protein = sum(foods_input[name]['protein'] * result_data.get(name, 0) for name in foods_input)
            cost = sum(foods_input[name]['cost'] * result_data.get(name, 0) for name in foods_input)
            
            # Tạo bảng tổng kết
            summary_data = {
                'Chỉ Số': ['Chi phí (¢)', 'Tổng Calo (kcal)', 'Calo từ chất béo (kcal)', 'Vitamin C (mg)', 'Protein (g)'],
                'Giá Trị Đạt Được': [f"{cost:.2f}", f"{total_cal:.2f}", f"{cal_fat:.2f}", f"{vit_c:.2f}", f"{protein:.2f}"],
                'Yêu Cầu Ràng Buộc Cố Định': [
                    'Minimize',
                    '400 - 600', 
                    f'<= 30% ({0.3 * total_cal:.2f})', 
                    f'>= 60', 
                    f'>= 12'
                ]
            }
            
            st.table(pd.DataFrame(summary_data))
            
        else:
            st.error(f"❌ KHÔNG TÌM THẤY LỜI GIẢI TỐI ƯU. **Trạng thái**: {result_data}. Vui lòng kiểm tra lại các ràng buộc hoặc dữ liệu nhập.")
        
    st.caption("Mô hình được giải quyết bằng PuLP (Integer Linear Programming).")

if __name__ == "__main__":
    main()

