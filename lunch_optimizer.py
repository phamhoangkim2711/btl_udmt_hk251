import streamlit as st
import pandas as pd
import pulp
from typing import List, Dict, Any

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(layout="wide", page_title="Tối Ưu Hóa Bữa Trưa (Custom Data Columns)")

# Định nghĩa kiểu dữ liệu cho ràng buộc tùy chỉnh
CustomConstraint = Dict[str, Any]

# --- DỮ LIỆU VÀ HÀM CỐ ĐỊNH ---

# Dữ liệu mặc định (Lưu ý: Chỉ dùng để khởi tạo, không giới hạn cột)
DEFAULT_FOODS_DATA = {
    'bread': {'cost': 5, 'cal_fat': 10, 'total_cal': 70, 'vit_c': 0, 'protein': 3},
    'peanut_butter': {'cost': 4, 'cal_fat': 75, 'total_cal': 100, 'vit_c': 0, 'protein': 4},
    'jelly': {'cost': 7, 'cal_fat': 0, 'total_cal': 50, 'vit_c': 3, 'protein': 0},
    'cracker': {'cost': 8, 'cal_fat': 20, 'total_cal': 60, 'vit_c': 0, 'protein': 1},
    'milk': {'cost': 15, 'cal_fat': 70, 'total_cal': 150, 'vit_c': 2, 'protein': 8},
    'juice': {'cost': 35, 'cal_fat': 0, 'total_cal': 100, 'vit_c': 120, 'protein': 1}
}

# DataFrame mặc định cho bảng ràng buộc
DEFAULT_CONSTRAINTS_DF = pd.DataFrame({
    'Nutrient': ['cost'],
    'Operator': ['<='],
    'Value': [200.0]
})

def run_optimization(foods_data: dict, custom_constraints: List[CustomConstraint]):
    """
    Hàm giải mô hình tối ưu hóa ăn trưa sử dụng PuLP, có thêm ràng buộc tùy chỉnh.
    """
    if not foods_data:
        return None, "Lỗi: Không có dữ liệu thực phẩm để chạy mô hình."

    food_names = list(foods_data.keys())
    model = pulp.LpProblem("Lunch Optimization Flexible", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("X", food_names, lowBound=0, cat='Integer')

    # Lấy danh sách các thuộc tính hợp lệ từ thực phẩm đầu tiên (để kiểm tra ràng buộc)
    valid_food_attributes = set(foods_data[food_names[0]].keys()) if food_names else set()

    # --- 1. HÀM MỤC TIÊU (Minimize Cost) ---
    if 'cost' not in valid_food_attributes:
        return None, "Lỗi: Dữ liệu thực phẩm phải có cột 'cost' để tối ưu hóa."

    model += (
        pulp.lpSum(foods_data[name]['cost'] * x[name] for name in food_names),
        "Total_Cost"
    )

    # --- 2. RÀNG BUỘC CỐ ĐỊNH (Chỉ thêm nếu cột tồn tại) ---
    
    # Ràng buộc Calo (C1, C2, C3)
    if 'total_cal' in valid_food_attributes and 'cal_fat' in valid_food_attributes:
        Total_Cal_Expr = pulp.lpSum(foods_data[name]['total_cal'] * x[name] for name in food_names)
        Cal_Fat_Expr = pulp.lpSum(foods_data[name]['cal_fat'] * x[name] for name in food_names)

        model += (Total_Cal_Expr >= 400, "Fixed_Min_Total_Calories")
        model += (Total_Cal_Expr <= 600, "Fixed_Max_Total_Calories")
        model += (Cal_Fat_Expr - 0.30 * Total_Cal_Expr <= 0, "Fixed_Max_30_Percent_Fat_Calories")

    # Ràng buộc Vitamin C (C4)
    if 'vit_c' in valid_food_attributes:
        model += (pulp.lpSum(foods_data[name]['vit_c'] * x[name] for name in food_names) >= 60, "Fixed_Min_Vitamin_C")

    # Ràng buộc Protein (C5)
    if 'protein' in valid_food_attributes:
        model += (pulp.lpSum(foods_data[name]['protein'] * x[name] for name in food_names) >= 12, "Fixed_Min_Protein")

    # RÀNG BUỘC ĐẶC BIỆT
    if 'bread' in food_names:
        model += (x['bread'] == 2, "Fixed_Exact_2_Slices_Bread")

    if 'peanut_butter' in food_names and 'jelly' in food_names:
        model += (x['peanut_butter'] - 2 * x['jelly'] >= 0, "Fixed_Peanut_Butter_vs_Jelly")

    liquid_items = [name for name in ['milk', 'juice'] if name in food_names]
    if liquid_items:
        model += (pulp.lpSum(x[name] for name in liquid_items) >= 1, "Fixed_Min_1_Cup_Liquid")

    # --- 3. RÀNG BUỘC TÙY CHỈNH (Custom Constraints) ---
    for i, constraint in enumerate(custom_constraints):
        nutrient = constraint.get('Nutrient', '').strip()
        operator = constraint.get('Operator', '').strip()
        value = constraint.get('Value', 0)

        # Kiểm tra tính hợp lệ
        if nutrient in valid_food_attributes and operator in ['>=', '<=', '=']:
            total_expr = pulp.lpSum(foods_data[name].get(nutrient, 0) * x[name] for name in food_names)
            
            # Thêm ràng buộc vào mô hình
            constraint_name = f"Custom_Constraint_{i+1}_{nutrient}_{operator}_{value}"
            if operator == '>=':
                model += (total_expr >= value, constraint_name)
            elif operator == '<=':
                model += (total_expr <= value, constraint_name)
            elif operator == '=':
                model += (total_expr == value, constraint_name)

    # --- 4. GIẢI MÔ HÌNH ---
    model.solve()

    if model.status == pulp.LpStatusOptimal:
        optimal_cost = pulp.value(model.objective)
        results = {name: int(round(x[name].varValue)) 
                   for name in food_names 
                   if x[name].varValue is not None and x[name].varValue > 1e-6}
        return optimal_cost, results
    
    return None, pulp.LpStatus[model.status]


# --- HÀM CHÍNH CỦA STREAMLIT ---
def main():
    st.title("🥪 TỐI ƯU HÓA CHI PHÍ BỮA TRƯA (Tùy chỉnh cột dữ liệu)")
    st.markdown("Bạn có thể thêm/xóa cột và hàng để định nghĩa các chất dinh dưỡng mới.")
    
    st.divider()

    ## 1. PHẦN NHẬP DỮ LIỆU THỰC PHẨM
    st.header("1. Nhập và chỉnh sửa dữ liệu thực phẩm")
    st.markdown("⚠️ **Lưu ý:**")
    st.markdown("* Cột **`cost`** là bắt buộc.")
    st.markdown("* Tên món ăn và tên cột cần viết liền không dấu (ví dụ: `fiber`, `vitamin_a`).")
    
    # --- CƠ CHẾ TÙY CHỈNH CỘT DỮ LIỆU ---
    
    # 1.1 Khởi tạo DataFrame có thể chỉnh sửa tên cột
    if 'editable_df' not in st.session_state:
        st.session_state.editable_df = pd.DataFrame.from_dict(DEFAULT_FOODS_DATA, orient='index')
        st.session_state.editable_df.index.name = 'food_name'

    # 1.2 Hiển thị data_editor cho phép thêm/xóa cột
    # Sử dụng `column_config` tự động
    
    col_config = {}
    for col in st.session_state.editable_df.columns:
         col_config[col] = st.column_config.NumberColumn(
                f"{col.replace('_', ' ').title()}", 
                min_value=0.0,
                format="%.2f"
            )

    edited_df = st.data_editor(
        st.session_state.editable_df,
        column_config=col_config,
        num_rows="dynamic", # Cho phép thêm/xóa hàng
        use_container_width=True,
        key="food_data_editor"
    )
    
    st.session_state.editable_df = edited_df.copy() # Cập nhật trạng thái

    # Chuyển DataFrame đã chỉnh sửa về dict cho PuLP
    foods_input = edited_df.to_dict('index')
    
    # Cập nhật danh sách thuộc tính hợp lệ sau khi người dùng chỉnh sửa
    if not edited_df.empty:
        valid_attributes = list(edited_df.columns)
    else:
        valid_attributes = []


    # --- KIỂM TRA LOGIC CƠ BẢN ---
    data_is_valid = True
    if 'cost' not in valid_attributes:
        st.error("❌ LỖI: Cột **`cost`** là bắt buộc để tối ưu hóa.")
        data_is_valid = False
    
    # Kiểm tra logic Calo nếu các cột tồn tại
    if 'cal_fat' in valid_attributes and 'total_cal' in valid_attributes:
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
    st.markdown(f"**Các chất dinh dưỡng hợp lệ:** `{', '.join(valid_attributes)}`")
    
    operator_options = ['>=', '<=', '=']

    custom_constraints_df = st.data_editor(
        DEFAULT_CONSTRAINTS_DF,
        column_config={
            "Nutrient": st.column_config.SelectboxColumn(
                "Chất dinh dưỡng",
                options=valid_attributes, # Tùy chỉnh danh sách dựa trên bảng thực phẩm
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

    if st.button("🚀 Chạy mô hình tối ưu", disabled=not data_is_valid or edited_df.empty):
        
        # Chạy mô hình PuLP
        optimal_cost, result_data = run_optimization(foods_input, custom_constraints)

        if optimal_cost is not None:
            st.success("✅ **ĐÃ TÌM THẤY KẾT QUẢ TỐI ƯU**")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Chi phí tối thiểu", f"{optimal_cost:.2f} ¢")
            
            solution_df = pd.DataFrame(
                result_data.items(), 
                columns=['Thực phẩm', 'Số lượng tối ưu']
            )
            solution_df['Số lượng tối ưu'] = solution_df['Số lượng tối ưu'].astype(int)
            
            with col2:
                st.dataframe(solution_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")

            # --- KIỂM TRA CÁC RÀNG BUỘC SAU KHI TỐI ƯU ---
            st.subheader("Kiểm tra Giá trị đạt được:")
            
            # Tính toán tất cả các thuộc tính của giải pháp tối ưu
            summary_data = {'Chỉ Số': [], 'Giá Trị Đạt Được': [], 'Ràng Buộc Cố Định (nếu có)': []}
            
            # Tính toán các giá trị đạt được
            for attr in valid_attributes:
                current_value = sum(foods_input[name].get(attr, 0) * result_data.get(name, 0) for name in foods_input)
                summary_data['Chỉ Số'].append(attr.replace('_', ' ').title())
                summary_data['Giá Trị Đạt Được'].append(f"{current_value:.2f}")

                # Thêm yêu cầu ràng buộc cố định (cho các cột cũ)
                fixed_req = 'N/A'
                if attr == 'cost': fixed_req = 'Minimize'
                elif attr == 'total_cal': fixed_req = '400 - 600'
                elif attr == 'cal_fat': fixed_req = f'<= 30% ({0.3 * current_value:.2f} kcal)' if 'total_cal' in valid_attributes else 'N/A'
                elif attr == 'vit_c': fixed_req = '>= 60 mg'
                elif attr == 'protein': fixed_req = '>= 12 g'
                
                summary_data['Ràng Buộc Cố Định (nếu có)'].append(fixed_req)

            # Thêm các ràng buộc tùy chỉnh để dễ kiểm tra
            for i, constraint in enumerate(custom_constraints):
                 nutrient = constraint.get('Nutrient', '').strip()
                 operator = constraint.get('Operator', '').strip()
                 value = constraint.get('Value', 0)
                 if nutrient and operator in ['>=', '<=', '='] and nutrient in valid_attributes:
                    # Tìm giá trị đã tính cho thuộc tính này
                    idx = valid_attributes.index(nutrient)
                    achieved_value = summary_data['Giá Trị Đạt Được'][idx]

                    summary_data['Chỉ Số'].append(f"Custom: {nutrient.replace('_', ' ').title()}")
                    summary_data['Giá Trị Đạt Được'].append(achieved_value)
                    summary_data['Ràng Buộc Cố Định (nếu có)'].append(f"{operator} {value:.2f}")

            st.table(pd.DataFrame(summary_data))
            
        else:
            st.error(f"❌ KHÔNG TÌM THẤY LỜI GIẢI TỐI ƯU. **Trạng thái**: {result_data}. Vui lòng kiểm tra lại các ràng buộc hoặc dữ liệu nhập.")
        
    st.caption("Mô hình được giải quyết bằng PuLP (Integer Linear Programming).")

if __name__ == "__main__":
    main()
