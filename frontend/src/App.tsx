import styled from 'styled-components';

const Wrapper = styled.div`
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #0b1f3a;
  color: #fefefe;
  font-family: 'Inter', sans-serif;
`;

function App() {
  return (
    <Wrapper>
      <h1>Wan Local Suite</h1>
    </Wrapper>
  );
}

export default App;
