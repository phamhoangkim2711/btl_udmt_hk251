import streamlit as st
import pandas as pd
import pulp
from typing import List, Dict, Any

# --- ĐỊNH NGHĨA KIỂU DỮ LIỆU ---
CustomConstraint = Dict[str, Any]
QuantityConstraint = Dict[str, Any]

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(layout="wide", page_title="Tối Ưu Hóa Bữa Trưa (Ổn định cột)")

# --- DỮ LIỆU MẶC ĐỊNH ---

# Dữ liệu mặc định thực phẩm
DEFAULT_FOODS_DICT = {
    'bread': {'cost': 5.0, 'cal_fat': 10.0, 'total_cal': 70.0, 'vit_c': 0.0, 'protein': 3.0},
    'peanut_butter': {'cost': 4.0, 'cal_fat': 75.0, 'total_cal': 100.0, 'vit_c': 0.0, 'protein': 4.0},
    'jelly': {'cost': 7.0, 'cal_fat': 0.0, 'total_cal': 50.0, 'vit_c': 3.0, 'protein': 0.0},
    'cracker': {'cost': 8.0, 'cal_fat': 20.0, 'total_cal': 60.0, 'vit_c': 0.0, 'protein': 1.0},
    'milk': {'cost': 15.0, 'cal_fat': 70.0, 'total_cal': 150.0, 'vit_c': 2.0, 'protein': 8.0},
    'juice': {'cost': 35.0, 'cal_fat': 0.0, 'total_cal': 100.0, 'vit_c': 120.0, 'protein': 1.0}
}

DEFAULT_COLUMNS = list(DEFAULT_FOODS_DICT['bread'].keys()) # cost, cal_fat, total_cal, vit_c, protein

# DataFrame mặc định cho ràng buộc tổng hợp
DEFAULT_CONSTRAINTS_DF = pd.DataFrame({
    'Nutrient': ['cost', 'protein'],
    'Operator': ['<=', '>='],
    'Value': [200.0, 15.0]
})

# DataFrame mặc định cho giới hạn số lượng
DEFAULT_QUANTITY_CONSTRAINTS_DF = pd.DataFrame({
    'Food_Name': ['bread'],
    'Min_Quantity': [2], 
    'Max_Quantity': [2]
})

# --- HÀM TỐI ƯU HÓA (PuLP) ---
def run_optimization(foods_data: dict, custom_constraints: List[CustomConstraint], quantity_constraints: List[QuantityConstraint]):
    """
    Hàm giải mô hình tối ưu hóa ăn trưa sử dụng PuLP.
    """
    if not foods_data:
        return None, "Lỗi: Không có dữ liệu thực phẩm để chạy mô hình."

    food_names = list(foods_data.keys())
    model = pulp.LpProblem("Lunch Optimization Flexible", pulp.LpMinimize)
    
    # 1. KHỞI TẠO BIẾN QUYẾT ĐỊNH
    x = pulp.LpVariable.dicts("X", food_names, lowBound=0, cat='Integer')

    # Lấy danh sách các thuộc tính hợp lệ
    valid_food_attributes = set(foods_data[food_names[0]].keys()) if food_names else set()

    # 2. RÀNG BUỘC SỐ LƯỢNG TỪ BẢNG NHẬP
    for constraint in quantity_constraints:
        food_name = constraint.get('Food_Name', '').strip()
        min_q = constraint.get('Min_Quantity', None)
        max_q = constraint.get('Max_Quantity', None)
        
        if food_name in food_names:
            var = x[food_name]
            
            # Cập nhật lowBound
            if isinstance(min_q, (int, float)) and min_q >= 0:
                var.lowBound = int(round(min_q)) 
                
            # Thêm ràng buộc Max_Quantity
            if isinstance(max_q, (int, float)) and max_q >= 0:
                model += (var <= max_q, f"Quantity_Max_{food_name}")

    # 3. HÀM MỤC TIÊU (Minimize Cost)
    if 'cost' not in valid_food_attributes:
        return None, "Lỗi: Dữ liệu thực phẩm phải có cột 'cost' để tối ưu hóa."

    model += (
        pulp.lpSum(foods_data[name]['cost'] * x[name] for name in food_names),
        "Total_Cost"
    )

    # 4. RÀNG BUỘC CỐ ĐỊNH 
    if 'total_cal' in valid_food_attributes and 'cal_fat' in valid_food_attributes:
        Total_Cal_Expr = pulp.lpSum(foods_data[name]['total_cal'] * x[name] for name in food_names)
        Cal_Fat_Expr = pulp.lpSum(foods_data[name]['cal_fat'] * x[name] for name in food_names)
        model += (Total_Cal_Expr >= 400, "Fixed_Min_Total_Calories")
        model += (Total_Cal_Expr <= 600, "Fixed_Max_Total_Calories")
        model += (Cal_Fat_Expr - 0.30 * Total_Cal_Expr <= 0, "Fixed_Max_30_Percent_Fat_Calories")
    
    if 'vit_c' in valid_food_attributes:
        model += (pulp.lpSum(foods_data[name]['vit_c'] * x[name] for name in food_names) >= 60, "Fixed_Min_Vitamin_C")

    if 'protein' in valid_food_attributes:
        model += (pulp.lpSum(foods_data[name]['protein'] * x[name] for name in food_names) >= 12, "Fixed_Min_Protein_Default")

    if 'peanut_butter' in food_names and 'jelly' in food_names:
        model += (x['peanut_butter'] - 2 * x['jelly'] >= 0, "Fixed_Peanut_Butter_vs_Jelly")

    liquid_items = [name for name in ['milk', 'juice'] if name in food_names]
    if liquid_items:
        model += (pulp.lpSum(x[name] for name in liquid_items) >= 1, "Fixed_Min_1_Cup_Liquid")

    # 5. RÀNG BUỘC TÙY CHỈNH
    for i, constraint in enumerate(custom_constraints):
        nutrient = constraint.get('Nutrient', '').strip()
        operator = constraint.get('Operator', '').strip()
        value = constraint.get('Value', 0)

        if nutrient in valid_food_attributes and operator in ['>=', '<=', '=']:
            total_expr = pulp.lpSum(foods_data[name].get(nutrient, 0) * x[name] for name in food_names)
            constraint_name = f"Custom_Constraint_{i+1}_{nutrient}_{operator}_{value}"
            if operator == '>=': model += (total_expr >= value, constraint_name)
            elif operator == '<=': model += (total_expr <= value, constraint_name)
            elif operator == '=': model += (total_expr == value, constraint_name)

    # 6. GIẢI MÔ HÌNH
    try:
        model.solve()
    except Exception as e:
        # Xử lý lỗi PuLP chung
        return None, f"Lỗi trong quá trình giải mô hình: {e}"

    if model.status == pulp.LpStatusOptimal:
        optimal_cost = pulp.value(model.objective)
        results = {name: int(round(x[name].varValue))
                   for name in food_names
                   if x[name].varValue is not None and x[name].varValue > 1e-6}
        return optimal_cost, results

    return None, pulp.LpStatus[model.status]


# --- HÀM CHÍNH CỦA STREAMLIT ---
def main():
    st.title("🥪 TỐI ƯU HÓA CHI PHÍ BỮA TRƯA")
    st.markdown("Định nghĩa các cột chất dinh dưỡng (tên) trước, sau đó nhập dữ liệu vào bảng.")

    st.divider()
    
    ## 1. PHẦN ĐỊNH NGHĨA CỘT VÀ NHẬP DỮ LIỆU
    st.header("1. Định nghĩa Cột và Dữ liệu Thực phẩm")
    
    # --- 1.1 Định nghĩa Cột (Chất dinh dưỡng) ---
    st.subheader("1.1 Định nghĩa Tên Cột Chất Dinh Dưỡng")
    st.markdown("Nhập tất cả tên cột bạn muốn sử dụng, cách nhau bởi dấu phẩy, **viết liền không dấu** (ví dụ: `cost, protein, fiber`). **`cost`** là cột bắt buộc.")

    if 'nutrient_columns_str' not in st.session_state:
        st.session_state.nutrient_columns_str = ', '.join(DEFAULT_COLUMNS)
    
    columns_str = st.text_input(
        "Danh sách tên cột (Chất dinh dưỡng):",
        value=st.session_state.nutrient_columns_str,
        key='nutrient_columns_input'
    )
    st.session_state.nutrient_columns_str = columns_str
    
    # Xử lý danh sách cột và kiểm tra lỗi
    input_columns = [col.strip() for col in columns_str.split(',') if col.strip()]
    if not input_columns or 'cost' not in input_columns:
        st.error("❌ Cột **`cost`** là bắt buộc và danh sách cột không được để trống.")
        return

    valid_attributes = ['cost'] + [col for col in input_columns if col != 'cost']
    st.markdown(f"**Các cột đang được sử dụng:** `{', '.join(valid_attributes)}`")

    # --- 1.2 Khởi tạo/Cập nhật DataFrame Thực phẩm ---
    
    # Logic khởi tạo DataFrame để đảm bảo cột ổn định
    if 'editable_df' not in st.session_state:
        st.session_state.editable_df = pd.DataFrame.from_dict(DEFAULT_FOODS_DICT, orient='index').rename_axis('food_name')
        
    # Tạo lại DataFrame với các cột mới và dữ liệu cũ
    current_data = st.session_state.editable_df.reset_index().to_dict('records')
    index_name = st.session_state.editable_df.index.name
    
    # Tạo DF mới chỉ với các cột hợp lệ
    new_df = pd.DataFrame(current_data).set_index(index_name) if current_data else pd.DataFrame(columns=[index_name])
    
    # Thêm cột mới và giữ lại thứ tự
    for col in valid_attributes:
        if col not in new_df.columns:
             new_df[col] = 0.0
    
    # Lọc lại các cột theo thứ tự định nghĩa
    new_df = new_df[[col for col in valid_attributes]]
    st.session_state.editable_df = new_df

    # --- 1.3 Hiển thị data_editor cho DỮ LIỆU ---
    st.subheader("1.2 Bảng dữ liệu Thực phẩm")
    st.markdown("Thêm/xóa hàng (món ăn) và nhập giá trị cho từng chất dinh dưỡng.")

    col_config = {}
    for col in valid_attributes:
          col_config[col] = st.column_config.NumberColumn(
              f"{col.replace('_', ' ').title()}",
              min_value=0.0,
              format="%.2f"
          )

    edited_df = st.data_editor(
        st.session_state.editable_df,
        column_config=col_config,
        num_rows="dynamic",
        use_container_width=True,
        key="food_data_editor"
    )

    st.session_state.editable_df = edited_df.copy()

    # --- BƯỚC LÀM SẠCH VÀ CHUYỂN ĐỔI DỮ LIỆU (KHẮC PHỤC LỖI NaN) ---
    cleaned_df = edited_df.copy()
    cleaned_df = cleaned_df[cleaned_df.index.notna()] # Loại bỏ hàng không có tên
    cleaned_df = cleaned_df.fillna(0.0) # Thay thế NaN bằng 0.0 (Quan trọng cho PuLP)
    
    foods_input = cleaned_df.to_dict('index')

    if not cleaned_df.empty:
        food_names = list(cleaned_df.index)
        data_is_valid = True
    else:
        food_names = []
        data_is_valid = False

    st.divider()

    ## 2. RÀNG BUỘC SỐ LƯỢNG VÀ RÀNG BUỘC TÙY CHỈNH
    st.header("2. Giới hạn Số lượng và Ràng Buộc Tùy Chỉnh")
    
    col_q, col_c = st.columns(2)
    
    with col_q:
        st.subheader("2.1 Giới hạn Số lượng Thực phẩm")
        
        # --- Bảng giới hạn số lượng ---
        if 'quantity_constraints_df' not in st.session_state:
            st.session_state.quantity_constraints_df = DEFAULT_QUANTITY_CONSTRAINTS_DF.copy()

        initial_q_data = []
        for name in food_names:
            existing_constraint = st.session_state.quantity_constraints_df[
                st.session_state.quantity_constraints_df['Food_Name'] == name
            ]
            if not existing_constraint.empty:
                initial_q_data.append(existing_constraint.iloc[0].to_dict())
            else:
                # Thiết lập mặc định cho món ăn mới
                min_q = 2 if name == 'bread' else 0
                max_q = 2 if name == 'bread' else 1000
                initial_q_data.append({'Food_Name': name, 'Min_Quantity': min_q, 'Max_Quantity': max_q})

        initial_q_df = pd.DataFrame(initial_q_data)

        quantity_constraints_df = st.data_editor(
            initial_q_df,
            column_config={
                "Food_Name": st.column_config.SelectboxColumn(
                    "Tên thực phẩm", options=food_names, required=True, disabled=True,
                ),
                "Min_Quantity": st.column_config.NumberColumn(
                    "Tối thiểu", min_value=0, format="%d", help="Số lượng tối thiểu (số nguyên)."
                ),
                "Max_Quantity": st.column_config.NumberColumn(
                    "Tối đa", min_value=0, format="%d", help="Số lượng tối đa (số nguyên)."
                )
            },
            num_rows="fixed",
            use_container_width=True,
            key="quantity_constraints_editor"
        )
        
        st.session_state.quantity_constraints_df = quantity_constraints_df.copy()
        quantity_constraints = quantity_constraints_df.to_dict('records')

    with col_c:
        st.subheader("2.2 Ràng Buộc Tổng Hợp")
        st.markdown(f"**Các chất dinh dưỡng hợp lệ:** `{', '.join(valid_attributes)}`")
        
        operator_options = ['>=', '<=', '=']

        custom_constraints_df = st.data_editor(
            DEFAULT_CONSTRAINTS_DF,
            column_config={
                "Nutrient": st.column_config.SelectboxColumn(
                    "Chất dinh dưỡng", options=valid_attributes, required=True,
                    help="Chọn thuộc tính của thực phẩm (Tên cột)."
                ),
                "Operator": st.column_config.SelectboxColumn(
                    "Toán tử", options=operator_options, required=True
                ),
                "Value": st.column_config.NumberColumn(
                    "Giá trị", min_value=0.0, format="%.2f", required=True
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

    if st.button("🚀 Chạy mô hình tối ưu", disabled=not data_is_valid or cleaned_df.empty):

        optimal_cost, result_data = run_optimization(foods_input, custom_constraints, quantity_constraints)

        if optimal_cost is not None and isinstance(result_data, dict):
            st.success("✅ **ĐÃ TÌM THẤY KẾT QUẢ TỐI ƯU**")

            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Chi phí tối thiểu", f"{optimal_cost:.2f} ¢")

            solution_df = pd.DataFrame(
                result_data.items(), columns=['Thực phẩm', 'Số lượng tối ưu']
            )
            solution_df['Số lượng tối ưu'] = solution_df['Số lượng tối ưu'].astype(int)
            with col2:
                st.dataframe(solution_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # --- KIỂM TRA CÁC RÀNG BUỘC SAU KHI TỐI ƯU ---
            st.subheader("Bảng tóm tắt Giá trị đạt được:")
            
            calculated_values = {}
            for attr in valid_attributes:
                current_value = sum(foods_input[name].get(attr, 0) * result_data.get(name, 0) for name in foods_input)
                calculated_values[attr] = current_value

            summary_list = []
            for attr in valid_attributes:
                fixed_req = 'N/A'
                if attr == 'cost': fixed_req = 'Minimize'
                elif attr == 'total_cal': fixed_req = '400 - 600'
                elif attr == 'cal_fat': 
                    if 'total_cal' in valid_attributes:
                        total_cal_achieved = calculated_values['total_cal']
                        fixed_req = f'<= 30% ({0.3 * total_cal_achieved:.2f})'
                elif attr == 'vit_c': fixed_req = '≥ 60.00'
                elif attr == 'protein': fixed_req = '≥ 12.00 (Mặc định)'

                summary_list.append({
                    'Chỉ Số': attr.replace('_', ' ').title(), 
                    'Giá Trị Đạt Được': f"{calculated_values.get(attr, 0):.2f}", 
                    'Ràng Buộc Cố Định/Mục Tiêu': fixed_req
                })

            for i, constraint in enumerate(custom_constraints):
                nutrient = constraint.get('Nutrient', '').strip()
                operator = constraint.get('Operator', '').strip()
                value = constraint.get('Value', 0)
                
                if nutrient in valid_attributes and operator in ['>=', '<=', '=']:
                     summary_list.append({
                        'Chỉ Số': f"**Custom: {nutrient.replace('_', ' ').title()}**",
                        'Giá Trị Đạt Được': f"{calculated_values.get(nutrient, 0):.2f}",
                        'Ràng Buộc Cố Định/Mục Tiêu': f"{operator} {value:.2f}"
                    })

            st.table(pd.DataFrame(summary_list))

        else:
            st.error(f"❌ KHÔNG TÌM THẤY LỜI GIẢI TỐI ƯU. **Trạng thái**: {result_data}. Vui lòng kiểm tra lại các ràng buộc hoặc dữ liệu nhập.")

    st.caption("Mô hình được giải quyết bằng PuLP (Integer Linear Programming).")

if __name__ == "__main__":
    main()



